/*
 * Copyright (c) 2002-2020 "Neo4j,"
 * Neo4j Sweden AB [http://neo4j.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package org.neo4j.cypher.internal.compiler.ast.convert.plannerQuery

import org.neo4j.cypher.internal.ast.RelationshipStartItem
import org.neo4j.cypher.internal.ast.semantics.SemanticTable
import org.neo4j.cypher.internal.compiler.helpers.ListSupport
import org.neo4j.cypher.internal.expressions.SemanticDirection
import org.neo4j.cypher.internal.ir.CallSubqueryHorizon
import org.neo4j.cypher.internal.ir.PatternRelationship
import org.neo4j.cypher.internal.ir.PlannerQueryPart
import org.neo4j.cypher.internal.ir.Predicate
import org.neo4j.cypher.internal.ir.QueryGraph
import org.neo4j.cypher.internal.ir.QueryHorizon
import org.neo4j.cypher.internal.ir.RegularSinglePlannerQuery
import org.neo4j.cypher.internal.ir.Selections
import org.neo4j.cypher.internal.ir.SimplePatternLength
import org.neo4j.cypher.internal.ir.SinglePlannerQuery
import org.neo4j.cypher.internal.ir.ordering.InterestingOrder
import org.neo4j.cypher.internal.util.NonEmptyList.IterableConverter
import org.neo4j.cypher.internal.util.UnNamedNameGenerator

import scala.collection.mutable

case class PlannerQueryBuilder(private val q: SinglePlannerQuery, semanticTable: SemanticTable)
  extends ListSupport {

  def amendQueryGraph(f: QueryGraph => QueryGraph): PlannerQueryBuilder =
    copy(q = q.updateTailOrSelf(_.amendQueryGraph(f)))

  def withHorizon(horizon: QueryHorizon): PlannerQueryBuilder =
    copy(q = q.updateTailOrSelf(_.withHorizon(horizon)))

  def withCallSubquery(subquery: PlannerQueryPart, correlated: Boolean): PlannerQueryBuilder = {
    withHorizon(CallSubqueryHorizon(subquery, correlated)).withTail(SinglePlannerQuery.empty)
  }

  def withTail(newTail: SinglePlannerQuery): PlannerQueryBuilder = {
    copy(q = q.updateTailOrSelf(_.withTail(newTail.amendQueryGraph(_.addArgumentIds(currentlyExposedSymbols.toIndexedSeq)))))
  }

  def withQueryInput(inputVariables : Seq[String]): PlannerQueryBuilder = {
    copy(q = q.withInput(inputVariables))
  }

  def withInterestingOrder(interestingOrder: InterestingOrder): PlannerQueryBuilder =
    copy(q = q.withTailInterestingOrder(interestingOrder))

  private def currentlyExposedSymbols: Set[String] = {
    q.lastQueryHorizon.exposedSymbols(q.lastQueryGraph.allCoveredIds)
  }

  def currentlyAvailableVariables: Set[String] = {
    val allPlannerQueries = q.allPlannerQueries
    val previousAvailableSymbols = if (allPlannerQueries.length > 1) {
      val current = allPlannerQueries(allPlannerQueries.length - 2)
      current.horizon.exposedSymbols(current.queryGraph.allCoveredIds)
    } else Set.empty

    // for the last planner query we should not consider the return projection
    previousAvailableSymbols ++ q.lastQueryGraph.allCoveredIds
  }

  def currentQueryGraph: QueryGraph = q.lastQueryGraph

  def allSeenPatternNodes: collection.Set[String] = {
    val nodes = mutable.Set[String]()

    val allPlannerQueries = q.allPlannerQueries
    if (allPlannerQueries.length > 1) {
      val current = allPlannerQueries(allPlannerQueries.length - 2)
      val projectedNodes = current.horizon.exposedSymbols(current.queryGraph.allCoveredIds).collect {
        case id@n if semanticTable.containsNode(n) => id
      }
      projectedNodes.foreach(nodes.add)
      current.queryGraph.collectAllPatternNodes(nodes.add)
    }
    q.lastQueryGraph.collectAllPatternNodes(nodes.add)
    nodes
  }

  def readOnly: Boolean = q.queryGraph.readOnly

  def build(): SinglePlannerQuery = {

    def fixArgumentIdsOnOptionalMatch(plannerQuery: SinglePlannerQuery): SinglePlannerQuery = {
      val optionalMatches = plannerQuery.queryGraph.optionalMatches
      val (_, newOptionalMatches) = optionalMatches.foldMap(plannerQuery.queryGraph.idsWithoutOptionalMatchesOrUpdates) {
        case (args, qg) =>
          (args ++ qg.allCoveredIds, qg.withArgumentIds(args intersect qg.dependencies))
      }
      plannerQuery
        .amendQueryGraph(_.withOptionalMatches(newOptionalMatches.toIndexedSeq))
        .updateTail(fixArgumentIdsOnOptionalMatch)
    }

    def fixArgumentIdsOnMerge(plannerQuery: SinglePlannerQuery): SinglePlannerQuery = {
      val newMergeMatchGraph = plannerQuery.queryGraph.mergeQueryGraph.map {
        qg =>
          val nodesAndRels = QueryGraph.coveredIdsForPatterns(qg.patternNodes, qg.patternRelationships)
          val predicateDependencies = qg.withoutArguments().dependencies
          val requiredArguments = nodesAndRels ++ predicateDependencies
          val availableArguments = qg.argumentIds
          qg.withArgumentIds(requiredArguments intersect availableArguments)
      }

      val updatePQ = newMergeMatchGraph match {
        case None =>
          plannerQuery
        case Some(qg) =>
          plannerQuery.amendQueryGraph(_.withMergeMatch(qg))
      }

      updatePQ.updateTail(fixArgumentIdsOnMerge)
    }

    def fixQueriesWithOnlyRelationshipIndex(plannerQuery: SinglePlannerQuery): SinglePlannerQuery = {
      val qg = plannerQuery.queryGraph
      val patternRelationships = qg.hints.collect {
        case r: RelationshipStartItem if !qg.patternRelationships.exists(_.name == r.name) =>
          val lNode = UnNamedNameGenerator.name(r.position)
          val rNode = UnNamedNameGenerator.name(r.position.bumped())

          PatternRelationship(r.name, (lNode, rNode), SemanticDirection.OUTGOING, Seq.empty, SimplePatternLength)
      }.toSeq

      val patternNodes = patternRelationships.flatMap(relationship => Set(relationship.nodes._1, relationship.nodes._2))
      plannerQuery
        .amendQueryGraph(_.addPatternRelationships(patternRelationships).addPatternNodes(patternNodes:_*))
        .updateTail(fixQueriesWithOnlyRelationshipIndex)
    }

    val fixedArgumentIds = q.foldMap {
      case (head, tail) =>
        val symbols = head.horizon.exposedSymbols(head.queryGraph.allCoveredIds)
        val newTailGraph = tail.queryGraph.withArgumentIds(symbols)
        tail.withQueryGraph(newTailGraph)
    }

    def groupInequalities(plannerQuery: SinglePlannerQuery): SinglePlannerQuery = {

      plannerQuery
        .amendQueryGraph(_.mapSelections {
          case Selections(predicates) =>
            val optPredicates = predicates.toNonEmptyListOption
            val newPredicates: Set[Predicate] = optPredicates.map { predicates =>
              groupInequalityPredicates(predicates).toSet
            }.getOrElse(predicates)
            Selections(newPredicates)
        })
        .updateTail(groupInequalities)
    }

    val withFixedOptionalMatchArgumentIds = fixArgumentIdsOnOptionalMatch(fixedArgumentIds)
    val withFixedMergeArgumentIds = fixArgumentIdsOnMerge(withFixedOptionalMatchArgumentIds)
    val groupedInequalities = groupInequalities(withFixedMergeArgumentIds)
    fixQueriesWithOnlyRelationshipIndex(groupedInequalities)
  }
}

object PlannerQueryBuilder {
  def apply(semanticTable: SemanticTable): PlannerQueryBuilder =
    PlannerQueryBuilder(SinglePlannerQuery.empty, semanticTable)
  def apply(semanticTable: SemanticTable, argumentIds: Set[String]): PlannerQueryBuilder =
    PlannerQueryBuilder(RegularSinglePlannerQuery(queryGraph = QueryGraph(argumentIds = argumentIds)), semanticTable)
}
