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
package org.neo4j.cypher.internal.frontend.phases

import org.neo4j.cypher.internal.expressions.NotEquals
import org.neo4j.cypher.internal.frontend.phases.CompilationPhaseTracer.CompilationPhase.AST_REWRITE
import org.neo4j.cypher.internal.rewriting.RewriterStepSequencer
import org.neo4j.cypher.internal.rewriting.conditions.containsNoNodesOfType
import org.neo4j.cypher.internal.rewriting.conditions.containsNoReturnAll
import org.neo4j.cypher.internal.rewriting.conditions.noDuplicatesInReturnItems
import org.neo4j.cypher.internal.rewriting.conditions.noReferenceEqualityAmongVariables
import org.neo4j.cypher.internal.rewriting.conditions.noUnnamedPatternElementsInMatch
import org.neo4j.cypher.internal.rewriting.conditions.noUnnamedPatternElementsInPatternComprehension
import org.neo4j.cypher.internal.rewriting.conditions.normalizedEqualsArguments
import org.neo4j.cypher.internal.rewriting.rewriters.InnerVariableNamer
import org.neo4j.cypher.internal.rewriting.rewriters.LiteralExtraction
import org.neo4j.cypher.internal.util.symbols.CypherType

case class AstRewriting(sequencer: String => RewriterStepSequencer,
                        literalExtraction: LiteralExtraction,
                        innerVariableNamer: InnerVariableNamer,
                        parameterTypeMapping : Map[String, CypherType] = Map.empty
) extends Phase[BaseContext, BaseState, BaseState] {

  private val astRewriter = new ASTRewriter(sequencer, literalExtraction, innerVariableNamer)

  override def process(in: BaseState, context: BaseContext): BaseState = {

    val (rewrittenStatement, extractedParams, _) = astRewriter.rewrite(in.statement(), in.semantics(), parameterTypeMapping, context.cypherExceptionFactory)

    in.withStatement(rewrittenStatement).withParams(extractedParams)
  }

  override def phase = AST_REWRITE

  override def description = "normalize the AST into a form easier for the planner to work with"

  override def postConditions: Set[Condition] = {
    val rewriterConditions = Set(
      noReferenceEqualityAmongVariables,
      noDuplicatesInReturnItems,
      containsNoReturnAll,
      noUnnamedPatternElementsInMatch,
      containsNoNodesOfType[NotEquals],
      normalizedEqualsArguments,
      noUnnamedPatternElementsInPatternComprehension
    )

    rewriterConditions.map(StatementCondition.apply)
  }
}
