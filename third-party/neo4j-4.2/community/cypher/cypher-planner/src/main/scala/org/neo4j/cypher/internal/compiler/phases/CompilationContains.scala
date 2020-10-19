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
package org.neo4j.cypher.internal.compiler.phases

import org.neo4j.cypher.internal.ast.Statement
import org.neo4j.cypher.internal.ast.semantics.SemanticState
import org.neo4j.cypher.internal.frontend.phases.Condition
import org.neo4j.cypher.internal.ir.UnionQuery
import org.neo4j.cypher.internal.logical.plans.LogicalPlan

import scala.reflect.ClassTag

case class CompilationContains[T: ClassTag](implicit manifest: Manifest[T]) extends Condition {

  override def check(in: AnyRef): Seq[String] = in match {
    case state: LogicalPlanState =>
      manifest.runtimeClass match {
        case x if classOf[Statement] == x && state.maybeStatement.isEmpty => Seq("Statement missing")
        case x if classOf[SemanticState] == x && state.maybeSemantics.isEmpty => Seq("Semantic State missing")
        case x if classOf[UnionQuery] == x && state.maybeQuery.isEmpty => Seq("Union query missing")
        case x if classOf[LogicalPlan] == x && state.maybeLogicalPlan.isEmpty => Seq("Logical plan missing")
        case _ => Seq.empty
      }
    case x => throw new IllegalArgumentException(s"Unknown state: $x")
  }
}
