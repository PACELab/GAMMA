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

import org.neo4j.cypher.internal.compiler.UpdateStrategy
import org.neo4j.cypher.internal.compiler.defaultUpdateStrategy
import org.neo4j.cypher.internal.compiler.planner.logical.steps.CandidateSelectorFactory
import org.neo4j.cypher.internal.compiler.planner.logical.steps.OptionalSolver
import org.neo4j.cypher.internal.compiler.planner.logical.steps.OrLeafPlanner
import org.neo4j.cypher.internal.compiler.planner.logical.steps.allNodesLeafPlanner
import org.neo4j.cypher.internal.compiler.planner.logical.steps.applyOptional
import org.neo4j.cypher.internal.compiler.planner.logical.steps.argumentLeafPlanner
import org.neo4j.cypher.internal.compiler.planner.logical.steps.idSeekLeafPlanner
import org.neo4j.cypher.internal.compiler.planner.logical.steps.indexScanLeafPlanner
import org.neo4j.cypher.internal.compiler.planner.logical.steps.indexSeekLeafPlanner
import org.neo4j.cypher.internal.compiler.planner.logical.steps.labelScanLeafPlanner
import org.neo4j.cypher.internal.compiler.planner.logical.steps.leftOuterHashJoin
import org.neo4j.cypher.internal.compiler.planner.logical.steps.pickBestPlanUsingHintsAndCost
import org.neo4j.cypher.internal.compiler.planner.logical.steps.rightOuterHashJoin
import org.neo4j.cypher.internal.compiler.planner.logical.steps.selectCovered
import org.neo4j.cypher.internal.compiler.planner.logical.steps.selectHasLabelWithJoin
import org.neo4j.cypher.internal.compiler.planner.logical.steps.selectPatternPredicates
import org.neo4j.cypher.internal.compiler.planner.logical.steps.triadicSelectionFinder
import org.neo4j.cypher.internal.ir.QueryGraph
import org.neo4j.cypher.internal.ir.ordering.InterestingOrder
import org.neo4j.cypher.internal.logical.plans.LogicalPlan

object QueryPlannerConfiguration {

  private val leafPlanFromExpressions: IndexedSeq[LeafPlanner with LeafPlanFromExpressions] = IndexedSeq(
    // MATCH (n) WHERE id(n) IN ... RETURN n
    idSeekLeafPlanner,

    // MATCH (n) WHERE n.prop IN ... RETURN n
    indexSeekLeafPlanner,

    // MATCH (n) WHERE has(n.prop) RETURN n
    // MATCH (n:Person) WHERE n.prop CONTAINS ...
    indexScanLeafPlanner,

    // MATCH (n:Person) RETURN n
    labelScanLeafPlanner
  )

  val allLeafPlanners = leafPlanFromExpressions ++ IndexedSeq(
    argumentLeafPlanner,

    // MATCH (n) RETURN n
    allNodesLeafPlanner,

    // Handles OR between other leaf planners
    OrLeafPlanner(leafPlanFromExpressions))


  val default: QueryPlannerConfiguration = {
    val predicateSelector = steps.Selector(pickBestPlanUsingHintsAndCost,
      selectPatternPredicates,
      triadicSelectionFinder,
      selectCovered,
      selectHasLabelWithJoin
    )


    QueryPlannerConfiguration(
      pickBestCandidate = pickBestPlanUsingHintsAndCost,
      applySelections = predicateSelector,
      optionalSolvers = Seq(
        applyOptional,
        leftOuterHashJoin,
        rightOuterHashJoin
      ),
      leafPlanners = LeafPlannerList(allLeafPlanners),
      updateStrategy = defaultUpdateStrategy
    )

  }
}

case class QueryPlannerConfiguration(leafPlanners: LeafPlannerIterable,
                                     applySelections: PlanSelector,
                                     optionalSolvers: Seq[OptionalSolver],
                                     pickBestCandidate: CandidateSelectorFactory,
                                     updateStrategy: UpdateStrategy) {

  def toKit(interestingOrder: InterestingOrder, context: LogicalPlanningContext) =
    QueryPlannerKit(
      select = (plan: LogicalPlan, qg: QueryGraph) => applySelections(plan, qg, interestingOrder, context),
      pickBest = pickBestCandidate(context),
      interestingOrder = interestingOrder
    )

  def withLeafPlanners(leafPlanners: LeafPlannerIterable): QueryPlannerConfiguration = copy(leafPlanners = leafPlanners)

  def withUpdateStrategy(updateStrategy: UpdateStrategy): QueryPlannerConfiguration = copy(updateStrategy = updateStrategy)
}

case class QueryPlannerKit(select: (LogicalPlan, QueryGraph) => LogicalPlan,
                           pickBest: CandidateSelector,
                           interestingOrder: InterestingOrder) {
  def select(plans: Iterable[Seq[LogicalPlan]], qg: QueryGraph): Iterable[Seq[LogicalPlan]] =
    plans.map(_.map(plan => select(plan, qg)))
}
