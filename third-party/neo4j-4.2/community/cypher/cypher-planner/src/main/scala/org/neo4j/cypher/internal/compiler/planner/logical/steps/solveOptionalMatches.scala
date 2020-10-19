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
package org.neo4j.cypher.internal.compiler.planner.logical.steps

import org.neo4j.cypher.internal.ast.Hint
import org.neo4j.cypher.internal.ast.UsingJoinHint
import org.neo4j.cypher.internal.compiler.planner.logical.LogicalPlanningContext
import org.neo4j.cypher.internal.compiler.planner.logical.plans.rewriter.unnestOptional
import org.neo4j.cypher.internal.ir.QueryGraph
import org.neo4j.cypher.internal.ir.ordering.InterestingOrder
import org.neo4j.cypher.internal.logical.plans.Argument
import org.neo4j.cypher.internal.logical.plans.LogicalPlan

trait OptionalSolver {
  def apply(qg: QueryGraph, lp: LogicalPlan, interestingOrder: InterestingOrder, context: LogicalPlanningContext): Option[LogicalPlan]
}

case object applyOptional extends OptionalSolver {
  override def apply(optionalQg: QueryGraph, lhs: LogicalPlan, interestingOrder: InterestingOrder, context: LogicalPlanningContext): Option[LogicalPlan] = {
    val innerContext: LogicalPlanningContext = context.withUpdatedCardinalityInformation(lhs)
    val inner = context.strategy.plan(optionalQg, interestingOrder, innerContext)
    val rhs = context.logicalPlanProducer.planOptional(inner, lhs.availableSymbols, innerContext)
    val applied = context.logicalPlanProducer.planApply(lhs, rhs, context)

    // Often the Apply can be rewritten into an OptionalExpand. We want to do that before cost estimating against the hash joins, otherwise that
    // is not a fair comparison (as they cannot be rewritten to something cheaper).
    Some(unnestOptional(applied).asInstanceOf[LogicalPlan])
  }
}

abstract class outerHashJoin extends OptionalSolver {
  override def apply(optionalQg: QueryGraph, side1: LogicalPlan, interestingOrder: InterestingOrder, context: LogicalPlanningContext): Option[LogicalPlan] = {
    val joinNodes = optionalQg.argumentIds
    val solvedHints = optionalQg.joinHints.filter { hint =>
      val hintVariables = hint.variables.map(_.name).toSet
      hintVariables.subsetOf(joinNodes)
    }

    // If side1 is just an Argument, any Apply above this will get written out so the incoming cardinality should be 1
    // This will be the case as [AssumeIndependenceQueryGraphCardinalityModel] will always use a cardinality of 1 if there are no
    // arguments and we delete the arguments below.
    // If not, then we're probably under an apply that will stay, so we need to force the cardinality to be multiplied by the incoming
    // cardinality.
    val side2Context = if (!side1.isInstanceOf[Argument]) context.copy(input = context.input.copy(alwaysMultiply = true)) else context

    val side2 = context.strategy.plan(optionalQg.withoutArguments().withoutHints(solvedHints.map(_.asInstanceOf[Hint])), interestingOrder, side2Context)

    if (joinNodes.nonEmpty &&
      joinNodes.forall(side1.availableSymbols) &&
      joinNodes.forall(optionalQg.patternNodes)) {
      Some(produceJoin(context, joinNodes, side1, side2, solvedHints))
    } else {
      None
    }
  }

  def produceJoin(context: LogicalPlanningContext, joinNodes: Set[String], side1: LogicalPlan, side2: LogicalPlan, solvedHints: Set[UsingJoinHint]): LogicalPlan
}

case object leftOuterHashJoin extends outerHashJoin {
  override def produceJoin(context: LogicalPlanningContext, joinNodes: Set[String], lhs: LogicalPlan, rhs: LogicalPlan, solvedHints: Set[UsingJoinHint]): LogicalPlan = {
    context.logicalPlanProducer.planLeftOuterHashJoin(joinNodes, lhs, rhs, solvedHints, context)
  }
}

case object rightOuterHashJoin extends outerHashJoin {
  override def produceJoin(context: LogicalPlanningContext, joinNodes: Set[String], rhs: LogicalPlan, lhs: LogicalPlan, solvedHints: Set[UsingJoinHint]): LogicalPlan = {
    context.logicalPlanProducer.planRightOuterHashJoin(joinNodes, lhs, rhs, solvedHints, context)
  }
}
