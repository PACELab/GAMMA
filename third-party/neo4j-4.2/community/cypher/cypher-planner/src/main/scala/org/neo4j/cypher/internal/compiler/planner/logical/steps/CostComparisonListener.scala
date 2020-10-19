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

import org.neo4j.cypher.internal.compiler.planner.logical.LogicalPlanningContext
import org.neo4j.cypher.internal.logical.plans.LogicalPlan

trait CostComparisonListener {
  def report[X](projector: X => LogicalPlan,
                input: Iterable[X],
                inputOrdering: Ordering[X],
                context: LogicalPlanningContext): Unit
}

object devNullListener extends CostComparisonListener {
  override def report[X](projector: X => LogicalPlan,
                         input: Iterable[X],
                         inputOrdering: Ordering[X],
                         context: LogicalPlanningContext): Unit = {}
}

object SystemOutCostLogger extends CostComparisonListener {
  def report[X](projector: X => LogicalPlan,
                input: Iterable[X],
                inputOrdering: Ordering[X],
                context: LogicalPlanningContext): Unit = {
    def stringTo(level: Int, plan: LogicalPlan): String = {
      def indent(level: Int, in: String): String = level match {
        case 0 => in
        case _ => System.lineSeparator() + "  " * level + in
      }

      val cost = context.cost(plan, context.input, context.planningAttributes.cardinalities)
      val thisPlan = indent(level, s"${plan.getClass.getSimpleName} costs $cost cardinality ${context.planningAttributes.cardinalities.get(plan.id)}")
      val l = plan.lhs.map(p => stringTo(level + 1, p)).getOrElse("")
      val r = plan.rhs.map(p => stringTo(level + 1, p)).getOrElse("")
      thisPlan + l + r
    }

    val sortedPlans = input.toIndexedSeq.sorted(inputOrdering).map(projector)

    if (sortedPlans.size > 1) {
      println("- Get best of:")
      for (plan <- sortedPlans) {

        val planTextWithCosts = stringTo(0, plan).replaceAll(System.lineSeparator(), System.lineSeparator() + "\t\t")
        val planText = plan.toString.replaceAll(System.lineSeparator(), System.lineSeparator() + "\t\t")
        println("=-" * 10)
        println(s"* Plan #${plan.debugId}")
        println(s"\t$planTextWithCosts")
        println(s"\t$planText")
        println(s"\t\tHints(${context.planningAttributes.solveds.get(plan.id).numHints})")
        println(s"\t\tlhs: ${plan.lhs}")
      }

      val best = sortedPlans.head
      println("!¡" * 10)
      println("- Best is:")
      println(s"Plan #${best.debugId}")
      println(s"\t${best.toString}")
      val planTextWithCosts = stringTo(0, best)
      println(s"\t$planTextWithCosts")
      println(s"\t\tHints(${context.planningAttributes.solveds.get(best.id).numHints})")
      println(s"\t\tlhs: ${best.lhs}")
      println("!¡" * 10)
      println()
    }
  }
}
