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
package org.neo4j.cypher.internal.compiler.planner.logical.cardinality.assumeIndependence

import org.neo4j.cypher.internal.ast.semantics.SemanticTable
import org.neo4j.cypher.internal.compiler.planner.logical.Metrics.QueryGraphCardinalityModel
import org.neo4j.cypher.internal.compiler.planner.logical.Metrics.QueryGraphSolverInput
import org.neo4j.cypher.internal.compiler.planner.logical.cardinality.ExpressionSelectivityCalculator
import org.neo4j.cypher.internal.compiler.planner.logical.cardinality.SelectivityCombiner
import org.neo4j.cypher.internal.compiler.planner.logical.cardinality.assumeIndependence.AssumeIndependenceQueryGraphCardinalityModel.MIN_INBOUND_CARDINALITY
import org.neo4j.cypher.internal.expressions.LabelName
import org.neo4j.cypher.internal.ir.QueryGraph
import org.neo4j.cypher.internal.ir.Selections
import org.neo4j.cypher.internal.ir.SimplePatternLength
import org.neo4j.cypher.internal.ir.VarPatternLength
import org.neo4j.cypher.internal.planner.spi.GraphStatistics
import org.neo4j.cypher.internal.util.Cardinality
import org.neo4j.cypher.internal.util.Selectivity

case class AssumeIndependenceQueryGraphCardinalityModel(stats: GraphStatistics, combiner: SelectivityCombiner)
  extends QueryGraphCardinalityModel {
  import AssumeIndependenceQueryGraphCardinalityModel.MAX_OPTIONAL_MATCH

  override val expressionSelectivityCalculator = ExpressionSelectivityCalculator(stats, combiner)
  private val patternSelectivityCalculator = PatternSelectivityCalculator(stats, combiner)

  /**
   * When there are optional matches, the cardinality is always the maximum of any matches that exist,
   * because no matches are limiting. So in principle we need to calculate cardinality of all possible combinations
   * of matches, and then take the max but since the number of combinations grow exponentially we cap it at a threshold.
   */
  def apply(queryGraph: QueryGraph, input: QueryGraphSolverInput, semanticTable: SemanticTable): Cardinality = {
    val combinations: Seq[QueryGraph] = findQueryGraphCombinations(queryGraph, input, semanticTable)

    val cardinalities = combinations.map(cardinalityForQueryGraph(_, input)(semanticTable))
    cardinalities.max
  }

  private def findQueryGraphCombinations(queryGraph: QueryGraph, input: QueryGraphSolverInput, semanticTable: SemanticTable): Seq[QueryGraph] =
    if (queryGraph.optionalMatches.isEmpty)
      Seq(queryGraph)
    else {
      //the number of combinations we need to consider grows exponentially, so above a threshold we only consider all
      //combinations of the most expensive query graphs
      val optionalMatches =
      if (queryGraph.optionalMatches.size <= MAX_OPTIONAL_MATCH) queryGraph.optionalMatches
      else queryGraph.optionalMatches.sortBy(-cardinalityForQueryGraph(_, input)(semanticTable).amount).take(MAX_OPTIONAL_MATCH)

      (0 to optionalMatches.length).flatMap(optionalMatches.combinations)
        .map(_.map(_.withoutArguments()))
        .map(_.foldLeft(QueryGraph.empty)(_.withOptionalMatches(Vector.empty) ++ _.withOptionalMatches(Vector.empty)))
        .map(queryGraph.withOptionalMatches(Vector.empty) ++ _)
    }

  private def calculateNumberOfPatternNodes(qg: QueryGraph) = {
    val intermediateNodes = qg.patternRelationships.map(_.length match {
      case SimplePatternLength            => 0
      case VarPatternLength(_, optMax)    => Math.max(Math.min(optMax.getOrElse(PatternSelectivityCalculator.MAX_VAR_LENGTH), PatternSelectivityCalculator.MAX_VAR_LENGTH) - 1, 0)
    }).sum

    qg.patternNodes.count(!qg.argumentIds.contains(_)) + intermediateNodes
  }

  private def cardinalityForQueryGraph(qg: QueryGraph, input: QueryGraphSolverInput)
                                      (implicit semanticTable: SemanticTable): Cardinality = {
    val (selectivity, numberOfZeroZeroRels) = calculateSelectivity(qg, input.labelInfo)
    val numberOfPatternNodes = calculateNumberOfPatternNodes(qg) - numberOfZeroZeroRels
    val numberOfGraphNodes = stats.nodesAllCardinality()

    // We can't always rely on arguments being present to indicate we need to multiply the cardinality
    // For example, when planning to solve an OPTIONAL MATCH with a join, we remove all the arguments. We
    // could still be beneath an Apply a this point though.
    val multiplier = if (input.alwaysMultiply || qg.argumentIds.nonEmpty) {
      Cardinality.max(input.inboundCardinality, MIN_INBOUND_CARDINALITY)
    } else {
      MIN_INBOUND_CARDINALITY
    }

    multiplier * (numberOfGraphNodes ^ numberOfPatternNodes) * selectivity
  }

  private def calculateSelectivity(qg: QueryGraph, labels: Map[String, Set[LabelName]])
                                  (implicit semanticTable: SemanticTable): (Selectivity, Int) = {
    implicit val selections: Selections = qg.selections

    val expressionSelectivities = selections.flatPredicates.map(expressionSelectivityCalculator(_))

    val patternSelectivities = qg.patternRelationships.toIndexedSeq.map {
      /* This is here to handle the *0..0 case.
         Our solution to the problem is to keep count of how many of these we see, and decrease the number of pattern
         nodes accordingly. The nice solution would have been to rewrite these relationships away at an earlier phase.
         This workaround should work, but might not give the best numbers.
       */
      case r if r.length == VarPatternLength(0, Some(0)) => None
      case r => Some(patternSelectivityCalculator(r, labels))
    }

    val numberOfZeroZeroRels = patternSelectivities.count(_.isEmpty)

    val selectivity = combiner.andTogetherSelectivities(expressionSelectivities ++ patternSelectivities.flatten)

    (selectivity.getOrElse(Selectivity.ONE), numberOfZeroZeroRels)
  }
}

object AssumeIndependenceQueryGraphCardinalityModel {
  //Since the number of combinations of optional matches grows exponentially and the reward of considering
  //more combination diminishes with the number of combinations, we cap it at this threshold.
  //The value chosen is mostly arbitrary but having it at 8 means 256 combinations which is still reasonably fast.
  private val MAX_OPTIONAL_MATCH = 8
  val MIN_INBOUND_CARDINALITY = Cardinality(1.0)
}
