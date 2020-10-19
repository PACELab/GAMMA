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
package org.neo4j.cypher.internal.compiler.planner.logical.debug

import org.neo4j.cypher.internal.ast.AliasedReturnItem
import org.neo4j.cypher.internal.ast.Query
import org.neo4j.cypher.internal.ast.Return
import org.neo4j.cypher.internal.ast.ReturnItems
import org.neo4j.cypher.internal.ast.SingleQuery
import org.neo4j.cypher.internal.ast.Statement
import org.neo4j.cypher.internal.compiler.phases.LogicalPlanState
import org.neo4j.cypher.internal.compiler.phases.PlannerContext
import org.neo4j.cypher.internal.expressions.ListLiteral
import org.neo4j.cypher.internal.expressions.StringLiteral
import org.neo4j.cypher.internal.expressions.Variable
import org.neo4j.cypher.internal.frontend.phases.CompilationPhaseTracer
import org.neo4j.cypher.internal.frontend.phases.CompilationPhaseTracer.CompilationPhase.LOGICAL_PLANNING
import org.neo4j.cypher.internal.frontend.phases.Condition
import org.neo4j.cypher.internal.frontend.phases.Phase
import org.neo4j.cypher.internal.logical.plans.Argument
import org.neo4j.cypher.internal.logical.plans.LogicalPlan
import org.neo4j.cypher.internal.logical.plans.ProduceResult
import org.neo4j.cypher.internal.logical.plans.UnwindCollection
import org.neo4j.cypher.internal.util.InputPosition
import org.neo4j.cypher.internal.util.attribution.SequentialIdGen

object DebugPrinter extends Phase[PlannerContext, LogicalPlanState, LogicalPlanState] {
  override def phase: CompilationPhaseTracer.CompilationPhase = LOGICAL_PLANNING

  override def description: String = "Print IR or AST as query result"

  override def process(from: LogicalPlanState, context: PlannerContext): LogicalPlanState = {
    val string = if (context.debugOptions.contains("querygraph"))
      from.query.toString
    else if (context.debugOptions.contains("ast"))
      from.statement().toString
    else if (context.debugOptions.contains("semanticstate"))
      from.semantics().toString
    else if (context.debugOptions.contains("logicalplan"))
      from.logicalPlan.toString
    else
      """Output options are: queryGraph, ast, semanticstate, logicalplan"""

    // We need to do this, otherwise the produced graph statistics won't work for creating an executable plan
    context.planContext.statistics.nodesAllCardinality()

    val (plan, newStatement) = stringToLogicalPlan(string)
    from.copy(maybePeriodicCommit = Some(None), maybeLogicalPlan = Some(plan), maybeStatement = Some(newStatement))
  }

  private def stringToLogicalPlan(string: String): (LogicalPlan, Statement) = {
    implicit val idGen = new SequentialIdGen()
    val pos = InputPosition(0, 0, 0)
    val stringValues = string.split(System.lineSeparator()).map(s => StringLiteral(s)(pos))
    val expression = ListLiteral(stringValues.toSeq)(pos)
    val unwind = UnwindCollection(Argument(Set.empty), "col", expression)
    val logicalPlan = ProduceResult(unwind, Seq("col"))

    val variable = Variable("col")(pos)
    val returnItem = AliasedReturnItem(variable, variable)(pos)
    val returnClause = Return(distinct = false, ReturnItems(includeExisting = false, Seq(returnItem))(pos), None, None, None, Set.empty)(pos)
    val newStatement = Query(None, SingleQuery(Seq(returnClause))(pos))(pos)

    (logicalPlan, newStatement)
  }

  override def postConditions: Set[Condition] = Set.empty
}
