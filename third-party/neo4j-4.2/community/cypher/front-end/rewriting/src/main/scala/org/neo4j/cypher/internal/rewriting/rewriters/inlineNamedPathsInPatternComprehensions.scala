/*
 * Copyright (c) 2002-2020 "Neo4j,"
 * Neo4j Sweden AB [http://neo4j.com]
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.neo4j.cypher.internal.rewriting.rewriters

import org.neo4j.cypher.internal.expressions.Expression
import org.neo4j.cypher.internal.expressions.LogicalVariable
import org.neo4j.cypher.internal.expressions.PathExpression
import org.neo4j.cypher.internal.expressions.PatternComprehension
import org.neo4j.cypher.internal.expressions.PatternElement
import org.neo4j.cypher.internal.util.Rewriter
import org.neo4j.cypher.internal.util.bottomUp

case object inlineNamedPathsInPatternComprehensions extends Rewriter {

  private val instance = bottomUp(Rewriter.lift {
    case expr @ PatternComprehension(Some(path), pattern, predicate, projection) =>
      val patternElement = pattern.element
      expr.copy(
        namedPath = None,
        predicate = predicate.map(_.inline(path, patternElement)),
        projection = projection.inline(path, patternElement)
      )(expr.position, expr.outerScope)
  })

  private implicit final class InliningExpression(val expr: Expression) extends AnyVal {
    def inline(path: LogicalVariable, patternElement: PatternElement) =
      expr.copyAndReplace(path) by {
        PathExpression(projectNamedPaths.patternPartPathExpression(patternElement))(expr.position)
      }
  }

  override def apply(v: AnyRef): AnyRef = instance(v)
}
