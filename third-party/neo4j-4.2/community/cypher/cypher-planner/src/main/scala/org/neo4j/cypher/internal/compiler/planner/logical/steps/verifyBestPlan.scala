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
import org.neo4j.cypher.internal.ast.UsingIndexHint
import org.neo4j.cypher.internal.ast.UsingJoinHint
import org.neo4j.cypher.internal.compiler.IndexHintUnfulfillableNotification
import org.neo4j.cypher.internal.compiler.JoinHintUnfulfillableNotification
import org.neo4j.cypher.internal.compiler.planner.logical.LogicalPlanningContext
import org.neo4j.cypher.internal.expressions.LabelName
import org.neo4j.cypher.internal.ir.PlannerQueryPart
import org.neo4j.cypher.internal.logical.plans.LogicalPlan
import org.neo4j.cypher.internal.planner.spi.PlanContext
import org.neo4j.exceptions.HintException
import org.neo4j.exceptions.IndexHintException
import org.neo4j.exceptions.InternalException
import org.neo4j.exceptions.JoinHintException

import scala.collection.JavaConverters.seqAsJavaListConverter

object verifyBestPlan {
  def apply(plan: LogicalPlan, expected: PlannerQueryPart, context: LogicalPlanningContext): Unit = {
    val constructed: PlannerQueryPart = context.planningAttributes.solveds.get(plan.id)

    if (expected != constructed) {
      val unfulfillableIndexHints = findUnfulfillableIndexHints(expected, context.planContext)
      val unfulfillableJoinHints = findUnfulfillableJoinHints(expected, context.planContext)
      val expectedWithoutHints = expected.withoutHints(unfulfillableIndexHints ++ unfulfillableJoinHints)
      if (expectedWithoutHints != constructed) {
        val a: PlannerQueryPart = expected.withoutHints(expected.allHints)
        val b: PlannerQueryPart = constructed.withoutHints(constructed.allHints)
        if (a != b) {
          // unknown planner issue failed to find plan (without regard for differences in hints)
          throw new InternalException(s"Expected \n$expected \n\n\nInstead, got: \n$constructed\nPlan: $plan")
        } else {
          // unknown planner issue failed to find plan matching hints (i.e. "implicit hints")
          val expectedHints = expected.allHints
          val actualHints = constructed.allHints
          val missing = expectedHints.diff(actualHints)
          val solvedInAddition = actualHints.diff(expectedHints)
          val inventedHintsAndThenSolvedThem = solvedInAddition.exists(!expectedHints.contains(_))
          if (missing.nonEmpty || inventedHintsAndThenSolvedThem) {
            def out(h: Set[Hint]) = h.mkString("`", ", ", "`")

            val details = if (missing.isEmpty)
              s"""Expected:
                 |${out(expectedHints)}
                 |
                 |Instead, got:
                 |${out(actualHints)}""".stripMargin
            else
              s"Could not solve these hints: ${out(missing)}"

            val message =
              s"""Failed to fulfil the hints of the query.
                 |$details
                 |
                 |Plan $plan""".stripMargin

            throw new HintException(message)
          }
        }
      } else {
        processUnfulfilledIndexHints(context, unfulfillableIndexHints)
        processUnfulfilledJoinHints(plan, context, unfulfillableJoinHints)
      }
    }
  }

  private def processUnfulfilledIndexHints(context: LogicalPlanningContext, hints: Set[UsingIndexHint]): Unit = {
    if (hints.nonEmpty) {
      // hints referred to non-existent indexes ("explicit hints")
      if (context.useErrorsOverWarnings) {
        val firstIndexHint = hints.head
        throw new IndexHintException(firstIndexHint.label.name, firstIndexHint.properties.map(_.name).asJava, "No such index")
      } else {
        hints.foreach { hint =>
          context.notificationLogger.log(IndexHintUnfulfillableNotification(hint.label.name, hint.properties.map(_.name)))
        }
      }
    }
  }

  private def processUnfulfilledJoinHints(plan: LogicalPlan, context: LogicalPlanningContext, hints: Set[UsingJoinHint]): Unit = {
    if (hints.nonEmpty) {
      // we were unable to plan hash join on some requested nodes
      if (context.useErrorsOverWarnings) {
        throw new JoinHintException(s"Unable to plan hash join. Instead, constructed\n$plan")
      } else {
        hints.foreach { hint =>
          context.notificationLogger.log(JoinHintUnfulfillableNotification(hint.variables.map(_.name).toIndexedSeq))
        }
      }
    }
  }

  private def findUnfulfillableIndexHints(query: PlannerQueryPart, planContext: PlanContext): Set[UsingIndexHint] = {
    query.allHints.flatMap {
      // using index name:label(property1,property2)
      case UsingIndexHint(_, LabelName(label), properties, _)
        if planContext.indexExistsForLabelAndProperties(label, properties.map(_.name)) => None
      // no such index exists
      case hint: UsingIndexHint => Some(hint)
      // don't care about other hints
      case _ => None
    }
  }

  private def findUnfulfillableJoinHints(query: PlannerQueryPart, planContext: PlanContext): Set[UsingJoinHint] = {
    query.allHints.collect {
      case hint: UsingJoinHint => hint
    }
  }
}
