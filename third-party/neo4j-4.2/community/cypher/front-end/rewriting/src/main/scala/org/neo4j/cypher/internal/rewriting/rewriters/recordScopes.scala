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

import org.neo4j.cypher.internal.ast.semantics.SemanticState
import org.neo4j.cypher.internal.expressions.ExistsSubClause
import org.neo4j.cypher.internal.expressions.MapProjection
import org.neo4j.cypher.internal.expressions.PatternComprehension
import org.neo4j.cypher.internal.util.Rewriter
import org.neo4j.cypher.internal.util.topDown

case class recordScopes(semanticState: SemanticState) extends Rewriter {

  def apply(that: AnyRef): AnyRef = instance.apply(that)

  private val instance: Rewriter = topDown(Rewriter.lift {
    case x: PatternComprehension =>
      x.withOuterScope(semanticState.recordedScopes(x).symbolDefinitions.map(_.asVariable))
    case x: ExistsSubClause =>
      x.withOuterScope(semanticState.recordedScopes(x).symbolDefinitions.map(_.asVariable))
    case x: MapProjection =>
      x.withDefinitionPos(semanticState.recordedScopes(x).symbolTable(x.name.name).definition.position)
  })
}
