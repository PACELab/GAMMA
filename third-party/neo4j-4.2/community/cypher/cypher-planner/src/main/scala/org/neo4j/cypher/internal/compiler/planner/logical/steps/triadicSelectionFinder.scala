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

import org.neo4j.cypher.internal.compiler.planner.logical.CandidateGenerator
import org.neo4j.cypher.internal.compiler.planner.logical.LogicalPlanningContext
import org.neo4j.cypher.internal.expressions.Ands
import org.neo4j.cypher.internal.expressions.Expression
import org.neo4j.cypher.internal.expressions.HasLabels
import org.neo4j.cypher.internal.expressions.NodePattern
import org.neo4j.cypher.internal.expressions.Not
import org.neo4j.cypher.internal.expressions.PatternExpression
import org.neo4j.cypher.internal.expressions.RelTypeName
import org.neo4j.cypher.internal.expressions.RelationshipChain
import org.neo4j.cypher.internal.expressions.RelationshipPattern
import org.neo4j.cypher.internal.expressions.RelationshipsPattern
import org.neo4j.cypher.internal.expressions.SemanticDirection
import org.neo4j.cypher.internal.expressions.Variable
import org.neo4j.cypher.internal.expressions.functions.Exists
import org.neo4j.cypher.internal.ir.QueryGraph
import org.neo4j.cypher.internal.ir.ordering.InterestingOrder
import org.neo4j.cypher.internal.logical.plans.Expand
import org.neo4j.cypher.internal.logical.plans.ExpandAll
import org.neo4j.cypher.internal.logical.plans.LogicalPlan
import org.neo4j.cypher.internal.logical.plans.Selection
import org.neo4j.cypher.internal.planner.spi.PlanningAttributes.Solveds
import org.neo4j.cypher.internal.util.attribution.SameId

object triadicSelectionFinder extends CandidateGenerator[LogicalPlan] {

  override def apply(in: LogicalPlan, qg: QueryGraph, interestingOrder: InterestingOrder, context: LogicalPlanningContext): Seq[LogicalPlan] =
    unsolvedPredicates(in, qg, context.planningAttributes.solveds).collect {
      // WHERE NOT (a)-[:X]->(c)
      case predicate@Not(Exists(patternExpr: PatternExpression)) => findMatchingRelationshipPattern(positivePredicate = false, predicate, patternExpr, in, qg, interestingOrder, context)
      // WHERE (a)-[:X]->(c)
      case predicate@Exists(patternExpr: PatternExpression) => findMatchingRelationshipPattern(positivePredicate = true, predicate, patternExpr, in, qg, interestingOrder, context)
    }.flatten

  def unsolvedPredicates(in: LogicalPlan, qg: QueryGraph, solveds: Solveds) = {
    val patternPredicates: Seq[Expression] = qg.selections.patternPredicatesGiven(in.availableSymbols)
    val solvedPredicates: Seq[Expression] = solveds.get(in.id).asSinglePlannerQuery.lastQueryGraph.selections.flatPredicates
    patternPredicates.filter { patternPredicate =>
      !(solvedPredicates contains patternPredicate)
    }
  }

  private def findMatchingRelationshipPattern(positivePredicate: Boolean,
                                              triadicPredicate: Expression,
                                              patternExpression: PatternExpression,
                                              in: LogicalPlan,
                                              qg: QueryGraph,
                                              interestingOrder: InterestingOrder,
                                              context: LogicalPlanningContext): Seq[LogicalPlan] = in match {

    // MATCH (a)-[:X]->(b)-[:X]->(c) WHERE (predicate involving (a)-[:X]->(c))
    case Selection(Ands(predicates),exp:Expand) => findMatchingOuterExpand(positivePredicate, triadicPredicate, patternExpression, predicates.toSeq, exp, qg, interestingOrder, context)

    // MATCH (a)-[:X]->(b)-[:Y]->(c) WHERE (predicate involving (a)-[:X]->(c))
    case exp:Expand => findMatchingOuterExpand(positivePredicate, triadicPredicate, patternExpression, Seq.empty, exp, qg, interestingOrder, context)

    case _ => Seq.empty
  }

  private def findMatchingOuterExpand(positivePredicate: Boolean,
                                      triadicPredicate: Expression,
                                      patternExpression: PatternExpression,
                                      incomingPredicates: Seq[Expression],
                                      expand: Expand,
                                      qg: QueryGraph,
                                      interestingOrder: InterestingOrder,
                                      context: LogicalPlanningContext): Seq[LogicalPlan] = expand match {
    case exp2@Expand(exp1: Expand, _, _, _, _, _, ExpandAll, _) =>
      findMatchingInnerExpand(positivePredicate, triadicPredicate, patternExpression, incomingPredicates, Seq.empty, exp1, exp2, qg, interestingOrder, context)

    case exp2@Expand(Selection(Ands(innerPredicates), exp1: Expand), _, _, _, _, _, ExpandAll, _) =>
      findMatchingInnerExpand(positivePredicate, triadicPredicate, patternExpression, incomingPredicates, innerPredicates.toSeq, exp1, exp2, qg, interestingOrder, context)

    case _ => Seq.empty
  }

  private def findMatchingInnerExpand(positivePredicate: Boolean,
                                      triadicPredicate: Expression,
                                      patternExpression: PatternExpression,
                                      incomingPredicates: Seq[Expression],
                                      leftPredicates: Seq[Expression],
                                      exp1: Expand,
                                      exp2: Expand,
                                      qg: QueryGraph,
                                      interestingOrder: InterestingOrder,
                                      context: LogicalPlanningContext): Seq[LogicalPlan] =
    if (exp1.mode == ExpandAll && exp1.to == exp2.from &&
      matchingLabels(positivePredicate, exp1.to, exp2.to, qg) &&
      leftPredicatesAcceptable(exp1.to, leftPredicates) &&
      matchingRelationshipPattern(patternExpression, exp1.from, exp2.to, exp1.types, exp1.dir)) {

      val left = if (leftPredicates.nonEmpty)
        context.logicalPlanProducer.planSelection(exp1, leftPredicates, context)
      else
        exp1

      val argument = context.logicalPlanProducer.planArgumentFrom(left, context)
      val newExpand2 = Expand(argument, exp2.from, exp2.dir, exp2.types, exp2.to, exp2.relName, ExpandAll)(SameId(exp2.id))
      val right = if (incomingPredicates.nonEmpty)
        context.logicalPlanProducer.planSelection(newExpand2, incomingPredicates, context)
      else
        newExpand2

      Seq(context.logicalPlanProducer.planTriadicSelection(positivePredicate, left, exp1.from, exp2.from, exp2.to, right, triadicPredicate, context))
    }
    else
      Seq.empty

  private def leftPredicatesAcceptable(leftId: String, leftPredicates: Seq[Expression]) = leftPredicates.forall {
    case HasLabels(Variable(id),List(_)) if id == leftId => true
    case a => false
  }

  private def matchingLabels(positivePredicate: Boolean, node1: String, node2: String, qg: QueryGraph): Boolean = {
    val labels1 = qg.selections.labelsOnNode(node1)
    val labels2 = qg.selections.labelsOnNode(node2)
    if (positivePredicate)
      labels1 == labels2
    else
      labels1.isEmpty || labels2.nonEmpty && (labels2 subsetOf labels1)
  }

  private def matchingRelationshipPattern(pattern: PatternExpression, from: String, to: String,
                                          types: Seq[RelTypeName], dir: SemanticDirection): Boolean = pattern match {
    // (a)-[:X]->(c)
    case PatternExpression(
    RelationshipsPattern(
    RelationshipChain(
    NodePattern(Some(Variable(predicateFrom)), List(), None, _),
    RelationshipPattern(None, predicateTypes, None, None, predicateDir, _, _),
    NodePattern(Some(Variable(predicateTo)), List(), None, _))))
      if predicateFrom == from && predicateTo == to && predicateTypes == types && predicateDir == dir => true
    case _ => false
  }
}
