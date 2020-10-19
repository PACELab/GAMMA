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

import org.neo4j.cypher.internal.expressions.FunctionInvocation
import org.neo4j.cypher.internal.expressions.GreaterThan
import org.neo4j.cypher.internal.expressions.NodePattern
import org.neo4j.cypher.internal.expressions.PatternExpression
import org.neo4j.cypher.internal.expressions.RelationshipChain
import org.neo4j.cypher.internal.expressions.RelationshipPattern
import org.neo4j.cypher.internal.expressions.RelationshipsPattern
import org.neo4j.cypher.internal.expressions.SignedDecimalIntegerLiteral
import org.neo4j.cypher.internal.expressions.functions
import org.neo4j.cypher.internal.logical.plans.NestedPlanExpression
import org.neo4j.cypher.internal.rewriting.rewriters.calculateUsingGetDegree
import org.neo4j.cypher.internal.util.Rewriter
import org.neo4j.cypher.internal.util.bottomUp

case object getDegreeRewriter extends Rewriter {

  def apply(that: AnyRef): AnyRef = instance(that)

  private val instance = bottomUp(rewriter,
    // Do not traverse into NestedPlanExpressions as they have been optimized already by an earlier call to plan
    _.isInstanceOf[NestedPlanExpression])

  private def rewriter = Rewriter.lift {
    // LENGTH( (a)-[]->() )
    case func@FunctionInvocation(_, _, _, IndexedSeq(PatternExpression(RelationshipsPattern(RelationshipChain(NodePattern(Some(node), List(), None, _), RelationshipPattern(None, types, None, None, dir, _, _), NodePattern(None, List(), None ,_))))))
      if func.function == functions.Length || func.function == functions.Size =>
      calculateUsingGetDegree(func, node, types, dir)

    // LENGTH( ()-[]->(a) )
    case func@FunctionInvocation(_, _, _, IndexedSeq(PatternExpression(RelationshipsPattern(RelationshipChain(NodePattern(None, List(), None, _), RelationshipPattern(None, types, None, None, dir, _, _), NodePattern(Some(node), List(), None, _))))))
      if func.function == functions.Length || func.function == functions.Size =>
      calculateUsingGetDegree(func, node, types, dir.reversed)

    // EXISTS( (a)-[]->() ) rewritten to GetDegree( (a)-[]->() ) > 0
    case func@FunctionInvocation(_, _, _, IndexedSeq(PatternExpression(RelationshipsPattern(RelationshipChain(NodePattern(Some(node), List(), None, _), RelationshipPattern(None, types, None, None, dir, _, _), NodePattern(None, List(), None, _))))))
      if func.function == functions.Exists  =>
      GreaterThan(calculateUsingGetDegree(func, node, types, dir), SignedDecimalIntegerLiteral("0")(func.position))(func.position)

    // EXISTS( ()-[]->(a) ) rewritten to GetDegree( (a)-[]->() ) > 0
    case func@FunctionInvocation(_, _, _, IndexedSeq(PatternExpression(RelationshipsPattern(RelationshipChain(NodePattern(None, List(), None, _), RelationshipPattern(None, types, None, None, dir, _, _), NodePattern(Some(node), List(), None, _))))))
      if func.function == functions.Exists =>
      GreaterThan(calculateUsingGetDegree(func, node, types, dir.reversed), SignedDecimalIntegerLiteral("0")(func.position))(func.position)
  }
}
