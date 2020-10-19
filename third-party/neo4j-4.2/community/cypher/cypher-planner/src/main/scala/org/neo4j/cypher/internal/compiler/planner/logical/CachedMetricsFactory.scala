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

import org.neo4j.cypher.internal.ast.semantics.SemanticTable
import org.neo4j.cypher.internal.compiler.CypherPlannerConfiguration
import org.neo4j.cypher.internal.compiler.helpers.CachedFunction
import org.neo4j.cypher.internal.compiler.planner.logical.Metrics.CardinalityModel
import org.neo4j.cypher.internal.compiler.planner.logical.Metrics.QueryGraphCardinalityModel
import org.neo4j.cypher.internal.compiler.planner.logical.cardinality.ExpressionSelectivityCalculator
import org.neo4j.cypher.internal.ir.PlannerQueryPart
import org.neo4j.cypher.internal.ir.QueryGraph
import org.neo4j.cypher.internal.planner.spi.GraphStatistics
import org.neo4j.cypher.internal.util.Cardinality

case class CachedMetricsFactory(metricsFactory: MetricsFactory) extends MetricsFactory {
  def newCardinalityEstimator(queryGraphCardinalityModel: QueryGraphCardinalityModel, evaluator: ExpressionEvaluator): CardinalityModel = {
    val wrapped: CardinalityModel = metricsFactory.newCardinalityEstimator(queryGraphCardinalityModel, evaluator)
    val cached = CachedFunction[PlannerQueryPart, Metrics.QueryGraphSolverInput, SemanticTable, Cardinality] { (a, b, c) => wrapped(a, b, c) }
    new CardinalityModel {
      override def apply(query: PlannerQueryPart, input: Metrics.QueryGraphSolverInput, semanticTable: SemanticTable): Cardinality = {
        cached.apply(query, input, semanticTable)
      }
    }
  }

  def newCostModel(config: CypherPlannerConfiguration) =
    CachedFunction(metricsFactory.newCostModel(config: CypherPlannerConfiguration))

  def newQueryGraphCardinalityModel(statistics: GraphStatistics): QueryGraphCardinalityModel = {
    val wrapped: QueryGraphCardinalityModel = metricsFactory.newQueryGraphCardinalityModel(statistics)
    val cached = CachedFunction[QueryGraph, Metrics.QueryGraphSolverInput, SemanticTable, Cardinality] { (a, b, c) => wrapped(a, b, c) }
    new QueryGraphCardinalityModel {
      override def apply(queryGraph: QueryGraph, input: Metrics.QueryGraphSolverInput, semanticTable: SemanticTable): Cardinality = {
        cached.apply(queryGraph, input, semanticTable)
      }

      override val expressionSelectivityCalculator: ExpressionSelectivityCalculator = wrapped.expressionSelectivityCalculator
    }
  }

}
