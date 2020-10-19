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
package org.neo4j.cypher.internal

import org.neo4j.cypher.internal.compiler.Neo4jCypherExceptionFactory
import org.neo4j.cypher.internal.parser.Base
import org.neo4j.cypher.internal.util.InputPosition
import org.parboiled.scala.Rule0
import org.parboiled.scala.Rule1
import org.parboiled.scala.group

final case class PreParsedStatement(statement: String, options: Seq[PreParserOption], offset: InputPosition)

case object CypherPreParser extends org.parboiled.scala.Parser with Base {
  def apply(input: String): PreParsedStatement = parseOrThrow(input, Neo4jCypherExceptionFactory(input, None), None, QueryWithOptions)

  def QueryWithOptions: Rule1[Seq[PreParsedStatement]] =
    WS ~ AllOptions ~ WS ~ AnySomething ~~>>
      ( (options: Seq[PreParserOption], text: String) => pos => Seq(PreParsedStatement(text, options, pos)))

  def AllOptions: Rule1[Seq[PreParserOption]] = zeroOrMore(AnyCypherOption, WS)

  def AnyCypherOption: Rule1[PreParserOption] = Cypher | Explain | Profile

  def AnySomething: Rule1[String] = rule("Query") { oneOrMore(org.parboiled.scala.ANY) ~> identity }

  def Cypher: Rule1[ConfigurationOptions] = rule("CYPHER options") {
    keyword("CYPHER") ~~
      optional(VersionNumber) ~~
      zeroOrMore(PlannerOption | RuntimeOption | ExpressionEngineOption | OperatorEngineOption | InterpretedPipesFallbackOption | ReplanOption | StrategyOption | DebugFlag, WS) ~~> ConfigurationOptions
  }

  def PlannerOption: Rule1[PreParserOption] = rule("planner option") (
    option("planner", "cost") ~ push(CostPlannerOption)
      | option("planner", "greedy") ~ push(GreedyPlannerOption)
      | option("planner", "idp") ~ push(IDPPlannerOption)
      | option("planner", "dp") ~ push(DPPlannerOption)
  )

  def RuntimeOption: Rule1[RuntimePreParserOption] = rule("runtime option")(
    option("runtime", "interpreted") ~ push(InterpretedRuntimeOption)
      | option("runtime", "slotted") ~ push(SlottedRuntimeOption)
      | option("runtime", "pipelined") ~ push(PipelinedRuntimeOption)
      | option("runtime", "parallel") ~ push(ParallelRuntimeOption)
  )

  def StrategyOption: Rule1[UpdateStrategyOption] = rule("strategy option")(
    option("updateStrategy", "eager") ~ push(EagerOption)
  )

  def VersionNumber: Rule1[VersionOption] = rule("Version") {
    group(Digits ~ "." ~ Digits) ~> VersionOption
  }

  def DebugFlag: Rule1[DebugOption] = rule("debug option") {
    keyword("debug") ~~ "=" ~~ SymbolicNameString ~~> DebugOption
  }

  def ExpressionEngineOption: Rule1[ExpressionEnginePreParserOption] = rule("expression engine option") (
    option("expressionEngine", "interpreted") ~ push(InterpretedExpressionOption)
      | option("expressionEngine", "compiled") ~ push(CompiledExpressionOption)
  )

  def OperatorEngineOption: Rule1[OperatorEnginePreParserOption] = rule("operator engine mode options") (
    option("operatorEngine", "compiled") ~ push(CompiledOperatorEngineOption)
      | option("operatorEngine", "interpreted") ~ push(InterpretedOperatorEngineOption)
  )

  def InterpretedPipesFallbackOption: Rule1[InterpretedPipesFallbackPreParserOption] = rule("interpreted pipes fallback options") (
    option("interpretedPipesFallback", "disabled") ~ push(DisabledInterpretedPipesFallbackOption)
      | option("interpretedPipesFallback", "default") ~ push(DefaultInterpretedPipesFallbackOption)
      | option("interpretedPipesFallback", "all") ~ push(AllInterpretedPipesFallbackOption)
  )

  def ReplanOption: Rule1[ReplanPreParserOption] = rule("replan strategy options") (
    option("replan", "force") ~ push(ReplanForceOption)
      | option("replan", "skip") ~ push(ReplanSkipOption)
      | option("replan", "default") ~ push(ReplanDefaultOption)
  )

  def Digits: Rule0 = oneOrMore("0" - "9")

  def Profile: Rule1[ExecutionModePreParserOption] = keyword("PROFILE") ~ push(ProfileOption)

  def Explain: Rule1[ExecutionModePreParserOption] = keyword("EXPLAIN") ~ push(ExplainOption)

  def option(key: String, value: String): Rule0 = {
    keyword(key) ~~ "=" ~~ keyword(value)
  }
}
