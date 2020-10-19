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
package org.neo4j.cypher.internal.compiler.planner.logical

import org.neo4j.cypher.internal.compiler.phases.LogicalPlanState
import org.neo4j.cypher.internal.compiler.phases.PlannerContext
import org.neo4j.cypher.internal.expressions.Equals
import org.neo4j.cypher.internal.expressions.Expression
import org.neo4j.cypher.internal.expressions.FunctionInvocation
import org.neo4j.cypher.internal.expressions.HasLabels
import org.neo4j.cypher.internal.expressions.LabelName
import org.neo4j.cypher.internal.expressions.LogicalVariable
import org.neo4j.cypher.internal.expressions.MapExpression
import org.neo4j.cypher.internal.expressions.NodePattern
import org.neo4j.cypher.internal.expressions.PatternExpression
import org.neo4j.cypher.internal.expressions.Property
import org.neo4j.cypher.internal.expressions.PropertyKeyName
import org.neo4j.cypher.internal.expressions.RelationshipChain
import org.neo4j.cypher.internal.expressions.RelationshipPattern
import org.neo4j.cypher.internal.expressions.RelationshipsPattern
import org.neo4j.cypher.internal.expressions.Variable
import org.neo4j.cypher.internal.frontend.phases.CompilationPhaseTracer.CompilationPhase
import org.neo4j.cypher.internal.frontend.phases.CompilationPhaseTracer.CompilationPhase.LOGICAL_PLANNING
import org.neo4j.cypher.internal.frontend.phases.Condition
import org.neo4j.cypher.internal.frontend.phases.Phase
import org.neo4j.cypher.internal.ir.AggregatingQueryProjection
import org.neo4j.cypher.internal.ir.DistinctQueryProjection
import org.neo4j.cypher.internal.ir.PatternRelationship
import org.neo4j.cypher.internal.ir.Predicate
import org.neo4j.cypher.internal.ir.QueryGraph
import org.neo4j.cypher.internal.ir.QueryProjection
import org.neo4j.cypher.internal.ir.RegularSinglePlannerQuery
import org.neo4j.cypher.internal.ir.Selections
import org.neo4j.cypher.internal.ir.SinglePlannerQuery
import org.neo4j.cypher.internal.ir.ordering.InterestingOrder
import org.neo4j.cypher.internal.util.InputPosition
import org.neo4j.cypher.internal.util.Rewritable.RewritableAny
import org.neo4j.cypher.internal.util.Rewriter
import org.neo4j.cypher.internal.util.topDown

import scala.annotation.tailrec
import scala.collection.TraversableOnce
import scala.collection.mutable
import scala.collection.mutable.ListBuffer

case object OptionalMatchRemover extends PlannerQueryRewriter {

  override def description: String = "remove optional match when possible"

  override def postConditions: Set[Condition] = Set.empty

  override def instance(ignored: PlannerContext): Rewriter = topDown(Rewriter.lift {
    case RegularSinglePlannerQuery(graph, interestingOrder, proj@AggregatingQueryProjection(distinctExpressions, aggregations, _, _), tail, queryInput)
      if validAggregations(aggregations) =>
      val projectionDeps: Iterable[LogicalVariable] = (distinctExpressions.values ++ aggregations.values).flatMap(_.dependencies)
      rewrite(projectionDeps, graph, interestingOrder, proj, tail, queryInput)

    case RegularSinglePlannerQuery(graph, interestingOrder, proj@DistinctQueryProjection(distinctExpressions, _, _), tail, queryInput) =>
      val projectionDeps: Iterable[LogicalVariable] = distinctExpressions.values.flatMap(_.dependencies)
      rewrite(projectionDeps, graph, interestingOrder, proj, tail, queryInput)
  })

  private def rewrite(projectionDeps: Iterable[LogicalVariable], graph: QueryGraph, interestingOrder: InterestingOrder, proj: QueryProjection, tail: Option[SinglePlannerQuery], queryInput: Option[Seq[String]]): RegularSinglePlannerQuery = {
    val updateDeps = graph.mutatingPatterns.flatMap(_.dependencies)
    val dependencies: Set[String] = projectionDeps.map(_.name).toSet ++ updateDeps
    val gen = new PositionGenerator

    val optionalMatches = graph.optionalMatches.flatMapWithTail {
      (original: QueryGraph, tail: Seq[QueryGraph]) =>

        //The dependencies on an optional match are:
        val allDeps =
        // dependencies from optional matches listed later in the query
          tail.flatMap(g => g.argumentIds ++ g.selections.variableDependencies).toSet ++
            // any dependencies from the next horizon
            dependencies --
            // But we don't need to solve variables already present by the non-optional part of the QG
            graph.idsWithoutOptionalMatchesOrUpdates

        val mustInclude = allDeps -- original.argumentIds
        val mustKeep = smallestGraphIncluding(original, mustInclude)

        if (mustKeep.isEmpty)
        // We did not find anything in this OPTIONAL MATCH. Since there are no variable deps from this clause,
        // and it can't change cardinality, it's safe to ignore it
          None
        else {
          val (predicatesForPatterns, remaining, elementsToKeep) = {
            val elementsToKeep1 = smallestGraphIncluding(original, mustInclude ++ original.argumentIds)
            extractElementsAndPatterns(original, mustInclude, elementsToKeep1)
          }

          val (patternsToKeep, patternsToFilter) = original.patternRelationships.partition(r => elementsToKeep(r.name))
          val patternNodes = original.patternNodes.filter(elementsToKeep.apply)

          val patternPredicates = patternsToFilter.map(toAst(elementsToKeep, predicatesForPatterns, gen, _))

          val newOptionalGraph = original.
            withPatternRelationships(patternsToKeep).
            withPatternNodes(patternNodes).
            withSelections(Selections.from(remaining) ++ patternPredicates)

          Some(newOptionalGraph)
        }
    }

    val matches = graph.withOptionalMatches(optionalMatches)
    RegularSinglePlannerQuery(matches, interestingOrder, horizon = proj, tail = tail, queryInput = queryInput)

  }

  private def extractElementsAndPatterns(original: QueryGraph, mustInclude: Set[String], elementsToKeepInitial: Set[String]):
  (Map[String, LabelsAndEquality], Set[Expression], Set[String]) = {
    val (predicatesForPatterns, remaining) =
      partitionPredicates(original.selections.predicates, elementsToKeepInitial)

    val variablesNeededForPredicates = remaining.flatMap(expression => expression.dependencies.map(_.name))
    val elementsToKeep = smallestGraphIncluding(original, mustInclude ++ original.argumentIds ++ variablesNeededForPredicates)

    if ( elementsToKeep.equals(elementsToKeepInitial) ) {
      (predicatesForPatterns, remaining, elementsToKeep)
    } else {
      extractElementsAndPatterns(original, mustInclude, elementsToKeep)
    }
  }

  private object LabelsAndEquality {
    def empty = new LabelsAndEquality(Seq.empty, Seq.empty)
  }

  private case class LabelsAndEquality(labels: Seq[LabelName], equality: Seq[(PropertyKeyName, Expression)])

  /**
   * This method extracts predicates that need to be part of pattern expressions
   *
   * @param predicates All the original predicates of the QueryGraph
   * @param kept       Set of all variables that should not be moved to pattern expressions
   * @return Map of label and property equality comparisons to move to pattern expressions,
   *         and the set of remaining predicates
   */
  private def partitionPredicates(predicates: Set[Predicate], kept: Set[String]): (Map[String, LabelsAndEquality], Set[Expression]) = {

    val patternPredicates = mutable.Map.empty[String, LabelsAndEquality]
    val predicatesToKeep = mutable.Set.empty[Expression]

    def addLabel(idName: String, labelName: LabelName) = {
      val current = patternPredicates.getOrElse(idName, LabelsAndEquality.empty)
      patternPredicates += idName -> current.copy(labels = current.labels :+ labelName)
    }

    def addProperty(idName: String, prop: PropertyKeyName, rhs: Expression) = {
      val current = patternPredicates.getOrElse(idName, LabelsAndEquality.empty)
      patternPredicates += idName -> current.copy(equality = current.equality :+ prop -> rhs)
    }

    predicates.foreach {
      case Predicate(deps, HasLabels(Variable(_), labels)) if deps.size == 1 && !kept(deps.head) =>
        require(labels.size == 1) // We know there is only a single label here because AST rewriting
        addLabel(deps.head, labels.head)
        ()

      case Predicate(deps, Equals(Property(Variable(_), prop), rhs)) if deps.size == 1 && !kept(deps.head) =>
        addProperty(deps.head, prop, rhs)
        ()

      case Predicate(_, expr) =>
        predicatesToKeep += expr
        ()
    }

    (patternPredicates.toMap, predicatesToKeep.toSet)
  }

  private def validAggregations(aggregations: Map[String, Expression]) =
    aggregations.isEmpty ||
      aggregations.values.forall {
        case func: FunctionInvocation => func.distinct
        case _ => false
      }

  private class PositionGenerator {
    private var pos: InputPosition = InputPosition.NONE

    def nextPosition(): InputPosition = {
      val current = pos
      //this is not nice but we want to make sure don't collide with "real positions"
      pos = pos.copy(offset = current.offset - 1)
      current
    }
  }

  private def toAst(elementsToKeep: Set[String], predicates: Map[String, LabelsAndEquality], gen: PositionGenerator, pattern: PatternRelationship) = {
    def createVariable(name: String): Option[Variable] =
      if (!elementsToKeep(name))
        None
      else {
        Some(Variable(name)(gen.nextPosition()))
      }

    def createNode(name: String): NodePattern = {
      val labelsAndProps = predicates.getOrElse(name, LabelsAndEquality.empty)
      val props = if (labelsAndProps.equality.isEmpty) None else Some(MapExpression(labelsAndProps.equality)(
        gen.nextPosition()))
      NodePattern(createVariable(name), labels = labelsAndProps.labels, properties = props)(gen.nextPosition())
    }

    val relName = createVariable(pattern.name)
    val leftNode = createNode(pattern.nodes._1)
    val rightNode = createNode(pattern.nodes._2)
    val relPattern = RelationshipPattern(relName, pattern.types, length = None, properties = None, pattern.dir)(
      gen.nextPosition())
    val chain = RelationshipChain(leftNode, relPattern, rightNode)(gen.nextPosition())
    PatternExpression(RelationshipsPattern(chain)(gen.nextPosition()))
  }

  implicit class FlatMapWithTailable(in: IndexedSeq[QueryGraph]) {
    def flatMapWithTail(f: (QueryGraph, Seq[QueryGraph]) => TraversableOnce[QueryGraph]): IndexedSeq[QueryGraph] = {

      @tailrec
      def recurse(that: QueryGraph, rest: Seq[QueryGraph], builder: mutable.Builder[QueryGraph, ListBuffer[QueryGraph]]): Unit = {
        builder ++= f(that, rest)
        if (rest.nonEmpty)
          recurse(rest.head, rest.tail, builder)
      }
      if (in.isEmpty)
        IndexedSeq.empty
      else {
        val builder = ListBuffer.newBuilder[QueryGraph]
        recurse(in.head, in.tail, builder)
        builder.result().toIndexedSeq
      }
    }
  }

  def smallestGraphIncluding(qg: QueryGraph, mustInclude: Set[String]): Set[String] = {
    if (mustInclude.size < 2)
      mustInclude intersect qg.allCoveredIds
    else {
      var accumulatedElements = mustInclude
      for {
        lhs <- mustInclude
        rhs <- mustInclude
        if lhs < rhs
      } {
        accumulatedElements ++= findPathBetween(qg, lhs, rhs)
      }
      accumulatedElements
    }
  }

  private case class PathSoFar(end: String, alreadyVisited: Set[PatternRelationship]) {
    def coveredIds: Set[String] = alreadyVisited.flatMap(_.coveredIds) + end
  }

  private def hasExpandedInto(from: Seq[PathSoFar], into: Seq[PathSoFar]): Seq[Set[String]] =
    for {lhs <- from
         rhs <- into
         if rhs.alreadyVisited.exists(p => p.coveredIds.contains(lhs.end))}
      yield {
        (lhs.alreadyVisited ++ rhs.alreadyVisited).flatMap(_.coveredIds)
      }


  private def expand(queryGraph: QueryGraph, from: Seq[PathSoFar]): Seq[PathSoFar] = {
    from.flatMap {
      case PathSoFar(end, alreadyVisited) =>
        queryGraph.patternRelationships.collect {
          case pr if !alreadyVisited(pr) && pr.coveredIds(end) =>
            PathSoFar(pr.otherSide(end), alreadyVisited + pr)
        }
    }
  }

  private def findPathBetween(qg: QueryGraph, startFromL: String, startFromR: String): Set[String] = {
    var l = Seq(PathSoFar(startFromL, Set.empty))
    var r = Seq(PathSoFar(startFromR, Set.empty))
    (0 to qg.patternRelationships.size) foreach { i =>
      if (i % 2 == 0) {
        l = expand(qg, l)
        val matches = hasExpandedInto(l, r)
        if (matches.nonEmpty)
          return matches.head
      }
      else {
        r = expand(qg, r)
        val matches = hasExpandedInto(r, l)
        if (matches.nonEmpty)
          return matches.head
      }
    }

    // Did not find any path. Let's do the safe thing and return everything
    qg.patternRelationships.flatMap(_.coveredIds)
  }


}

trait PlannerQueryRewriter extends Phase[PlannerContext, LogicalPlanState, LogicalPlanState] {
  override def phase: CompilationPhase = LOGICAL_PLANNING

  def instance(context: PlannerContext): Rewriter

  override def process(from: LogicalPlanState, context: PlannerContext): LogicalPlanState = {
    val query = from.query
    val rewritten = query.endoRewrite(instance(context))
    from.copy(maybeQuery = Some(rewritten))
  }
}
