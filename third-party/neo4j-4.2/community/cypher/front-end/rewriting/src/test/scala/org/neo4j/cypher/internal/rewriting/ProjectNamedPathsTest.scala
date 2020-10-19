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
package org.neo4j.cypher.internal.rewriting

import org.neo4j.cypher.internal.ast.AliasedReturnItem
import org.neo4j.cypher.internal.ast.AscSortItem
import org.neo4j.cypher.internal.ast.Match
import org.neo4j.cypher.internal.ast.OrderBy
import org.neo4j.cypher.internal.ast.Query
import org.neo4j.cypher.internal.ast.Return
import org.neo4j.cypher.internal.ast.ReturnItems
import org.neo4j.cypher.internal.ast.SingleQuery
import org.neo4j.cypher.internal.ast.Where
import org.neo4j.cypher.internal.ast.With
import org.neo4j.cypher.internal.ast.semantics.SemanticState
import org.neo4j.cypher.internal.expressions.CountStar
import org.neo4j.cypher.internal.expressions.EveryPath
import org.neo4j.cypher.internal.expressions.MultiRelationshipPathStep
import org.neo4j.cypher.internal.expressions.NilPathStep
import org.neo4j.cypher.internal.expressions.NodePathStep
import org.neo4j.cypher.internal.expressions.NodePattern
import org.neo4j.cypher.internal.expressions.PathExpression
import org.neo4j.cypher.internal.expressions.Pattern
import org.neo4j.cypher.internal.expressions.RelationshipChain
import org.neo4j.cypher.internal.expressions.RelationshipPattern
import org.neo4j.cypher.internal.expressions.SemanticDirection
import org.neo4j.cypher.internal.expressions.SingleRelationshipPathStep
import org.neo4j.cypher.internal.expressions.Variable
import org.neo4j.cypher.internal.rewriting.rewriters.expandStar
import org.neo4j.cypher.internal.rewriting.rewriters.normalizeWithAndReturnClauses
import org.neo4j.cypher.internal.rewriting.rewriters.projectNamedPaths
import org.neo4j.cypher.internal.util.OpenCypherExceptionFactory
import org.neo4j.cypher.internal.util.inSequence
import org.neo4j.cypher.internal.util.test_helpers.CypherFunSuite

class ProjectNamedPathsTest extends CypherFunSuite with AstRewritingTestSupport {

  private def projectionInlinedAst(queryText: String) = ast(queryText).endoRewrite(projectNamedPaths)

  private def ast(queryText: String) = {
    val parsed = parser.parse(queryText, OpenCypherExceptionFactory(None))
    val exceptionFactory = OpenCypherExceptionFactory(Some(pos))
    val normalized = parsed.endoRewrite(inSequence(normalizeWithAndReturnClauses(exceptionFactory)))
    val checkResult = normalized.semanticCheck(SemanticState.clean)
    normalized.endoRewrite(inSequence(expandStar(checkResult.state)))
  }

  private def parseReturnedExpr(queryText: String) =
    projectionInlinedAst(queryText) match {
      case Query(_, SingleQuery(Seq(_, Return(_, ReturnItems(_, Seq(AliasedReturnItem(expr, Variable("p")))), _, _, _, _)))) => expr
    }

  test("MATCH p = (a) RETURN p" ) {
    val returns = parseReturnedExpr("MATCH p = (a) RETURN p")

    val expected = PathExpression(
      NodePathStep(varFor("a"), NilPathStep)
    )_

    returns should equal(expected: PathExpression)
  }

  test("MATCH p = (a) WITH p RETURN p" ) {
    val rewritten = projectionInlinedAst("MATCH p = (a) WITH p RETURN p")
    val a = varFor("a")
    val p = varFor("p")
    val MATCH =
      Match(optional = false,
        Pattern(List(
          EveryPath(
            NodePattern(Some(a), List(), None)(pos))
        ))(pos), List(), None)(pos)

    val WITH =
      With(distinct = false,
        ReturnItems(includeExisting = false, Seq(
          AliasedReturnItem(PathExpression(NodePathStep(a, NilPathStep))(pos), p)(pos)
        ))(pos), None, None, None, None)(pos)

    val RETURN =
      Return(distinct = false,
        ReturnItems(includeExisting = false, Seq(
          AliasedReturnItem(p, p)(pos)
        ))(pos), None, None, None)(pos)

    val expected: Query = Query(None, SingleQuery(List(MATCH, WITH, RETURN))(pos))(pos)

    rewritten should equal(expected)
  }

  //don't project what is already projected
  test("MATCH p = (a) WITH p, a RETURN p" ) {
    val rewritten = projectionInlinedAst("MATCH p = (a) WITH p, a RETURN p")
    val a = varFor("a")
    val p = varFor("p")
    val MATCH =
      Match(optional = false,
        Pattern(List(
          EveryPath(
            NodePattern(Some(a), List(), None)(pos))
        ))(pos), List(), None)(pos)

    val WITH =
      With(distinct = false,
        ReturnItems(includeExisting = false, Seq(
          AliasedReturnItem(PathExpression(NodePathStep(a, NilPathStep))(pos), p)(pos),
          AliasedReturnItem(a, a)(pos)
        ))(pos), None, None, None, None)(pos)

    val RETURN=
      Return(distinct = false,
        ReturnItems(includeExisting = false, Seq(
          AliasedReturnItem(p, p)(pos)
        ))(pos), None, None, None)(pos)

    val expected: Query = Query(None, SingleQuery(List(MATCH, WITH, RETURN))(pos))(pos)

    rewritten should equal(expected)
  }

  test("MATCH p = (a) WITH p MATCH q = (b) RETURN p, q" ) {
    val rewritten = projectionInlinedAst("MATCH p = (a) WITH p MATCH q = (b) WITH p, q RETURN p, q")
    val a = varFor("a")
    val b = varFor("b")
    val p = varFor("p")
    val q = varFor("q")

    val MATCH1 =
      Match(optional = false,
        Pattern(List(
          EveryPath(
            NodePattern(Some(a), List(), None)(pos))
        ))(pos), List(), None)(pos)

    val WITH1 =
      With(distinct = false,
        ReturnItems(includeExisting = false, Seq(
          AliasedReturnItem(PathExpression(NodePathStep(a, NilPathStep))(pos), p)(pos)
        ))(pos), None, None, None, None)(pos)

    val MATCH2 =
      Match(optional = false,
        Pattern(List(
          EveryPath(
            NodePattern(Some(b), List(), None)(pos))
        ))(pos), List(), None)(pos)

    val WITH2 =
      With(distinct = false,
        ReturnItems(includeExisting = false, Seq(
          AliasedReturnItem(p, p)(pos),
          AliasedReturnItem(PathExpression(NodePathStep(b, NilPathStep))(pos), q)(pos)
        ))(pos), None, None, None, None)(pos)

    val RETURN=
      Return(distinct = false,
        ReturnItems(includeExisting = false, Seq(
          AliasedReturnItem(p, p)(pos),
          AliasedReturnItem(q, q)(pos)
        ))(pos), None, None, None)(pos)

    val expected: Query = Query(None, SingleQuery(List(MATCH1, WITH1, MATCH2, WITH2, RETURN))(pos))(pos)

    rewritten should equal(expected)
  }

  test("MATCH p = (a)-[r]->(b) RETURN p" ) {
    val returns = parseReturnedExpr("MATCH p = (a)-[r]->(b) RETURN p")

    val expected = PathExpression(
      NodePathStep(varFor("a"), SingleRelationshipPathStep(varFor("r"), SemanticDirection.OUTGOING, Some(varFor("b")), NilPathStep))
    )_

    returns should equal(expected: PathExpression)
  }

  test("MATCH p = (b)<-[r]->(a) RETURN p" ) {
    val returns = parseReturnedExpr("MATCH p = (b)<-[r]-(a) RETURN p")

    val expected = PathExpression(
      NodePathStep(varFor("b"), SingleRelationshipPathStep(varFor("r"), SemanticDirection.INCOMING, Some(varFor("a")), NilPathStep))
    )_

    returns should equal(expected: PathExpression)
  }

  test("MATCH p = (a)-[r*]->(b) RETURN p" ) {
    val returns = parseReturnedExpr("MATCH p = (a)-[r*]->(b) RETURN p")

    val expected = PathExpression(
      NodePathStep(varFor("a"), MultiRelationshipPathStep(varFor("r"), SemanticDirection.OUTGOING, Some(varFor("b")), NilPathStep))
    )_

    returns should equal(expected: PathExpression)
  }

  test("MATCH p = (b)<-[r*]-(a) RETURN p AS p" ) {
    val returns = parseReturnedExpr("MATCH p = (b)<-[r*]-(a) RETURN p AS p")

    val expected = PathExpression(
      NodePathStep(varFor("b"), MultiRelationshipPathStep(varFor("r"), SemanticDirection.INCOMING, Some(varFor("a")), NilPathStep))
    )_

    returns should equal(expected: PathExpression)
  }

  test("MATCH p = (a)-[r]->(b) RETURN p, 42 as order ORDER BY order") {
    val rewritten = projectionInlinedAst("MATCH p = (a)-[r]->(b) RETURN p, 42 as order ORDER BY order")

    val aId = varFor("a")
    val orderId = varFor("order")
    val rId = varFor("r")
    val pId = varFor("p")
    val bId = varFor("b")

    val MATCH =
      Match(optional = false,
        Pattern(List(
          EveryPath(
            RelationshipChain(
              NodePattern(Some(aId), List(), None)(pos),
              RelationshipPattern(Some(rId), List(), None, None, SemanticDirection.OUTGOING)(pos), NodePattern(Some(bId), List(), None)(pos)
            )(pos))
        ))(pos), List(), None)(pos)

    val RETURN =
      Return(distinct = false,
        ReturnItems(includeExisting = false, Seq(
          AliasedReturnItem(PathExpression(NodePathStep(aId, SingleRelationshipPathStep(rId, SemanticDirection.OUTGOING, Some(varFor("b")), NilPathStep)))(pos), pId)(pos),
          AliasedReturnItem(literalInt(42), orderId)(pos)
        ))(pos),
        Some(OrderBy(List(AscSortItem(orderId)(pos)))(pos)),
        None, None
      )(pos)

    val expected: Query = Query(None, SingleQuery(List(MATCH, RETURN))(pos))(pos)

    rewritten should equal(expected)
  }

  test("MATCH p = (a)-[r]->(b) WHERE length(p) > 10 RETURN 1") {
    val rewritten = projectionInlinedAst("MATCH p = (a)-[r]->(b) WHERE length(p) > 10 RETURN 1 as x")

    val aId = varFor("a")
    val rId = varFor("r")
    val bId = varFor("b")

    val WHERE =
      Where(
        greaterThan(
          function("length", PathExpression(NodePathStep(aId, SingleRelationshipPathStep(rId, SemanticDirection.OUTGOING, Some(varFor("b")), NilPathStep)))(pos)),
          literalInt(10)
        )
      )(pos)

    val MATCH =
      Match(optional = false,
        Pattern(List(
          EveryPath(
            RelationshipChain(
              NodePattern(Some(aId), List(), None)(pos),
              RelationshipPattern(Some(rId), List(), None, None, SemanticDirection.OUTGOING)(pos), NodePattern(Some(bId), List(), None)(pos)
            )(pos))
        ))(pos), List(), Some(WHERE))(pos)

    val RETURN =
      Return(distinct = false,
        ReturnItems(includeExisting = false, List(
          AliasedReturnItem(literalInt(1), varFor("x"))(pos)
        ))(pos),
        None, None, None
      )(pos)

    val expected: Query = Query(None, SingleQuery(List(MATCH, RETURN))(pos))(pos)

    rewritten should equal(expected)
  }

  test("Aggregating WITH downstreams" ) {
    val rewritten = projectionInlinedAst("MATCH p = (a) WITH length(p) as l, count(*) as x WITH l, x RETURN l + x")
    val a = varFor("a")
    val l = varFor("l")
    val x = varFor("x")
    val MATCH =
      Match(optional = false,
        Pattern(List(
          EveryPath(
            NodePattern(Some(a), List(), None)(pos))
        ))(pos), List(), None)(pos)

    val pathExpression = PathExpression(NodePathStep(a, NilPathStep))(pos)
    val WITH1 =
      With(distinct = false,
        ReturnItems(includeExisting = false, Seq(
          AliasedReturnItem(function("length", pathExpression), l)(pos),
          AliasedReturnItem(CountStar()(pos), x)(pos)
        ))(pos), None, None, None, None)(pos)

    val WITH2 =
      With(distinct = false,
        ReturnItems(includeExisting = false, Seq(
          AliasedReturnItem(l, l)(pos),
          AliasedReturnItem(x, x)(pos)
        ))(pos), None, None, None, None)(pos)

    val RETURN =
      Return(distinct = false,
        ReturnItems(includeExisting = false, Seq(
          AliasedReturnItem(add(l, x), varFor("l + x"))(pos)
        ))(pos), None, None, None)(pos)

    val expected: Query = Query(None, SingleQuery(List(MATCH, WITH1, WITH2, RETURN))(pos))(pos)

    rewritten should equal(expected)
  }

  test("WHERE and ORDER BY on WITH clauses should be rewritten" ) {
    val rewritten = projectionInlinedAst("MATCH p = (a) WITH a ORDER BY p WHERE length(p) = 1 RETURN a")

    val aId = varFor("a")

    val MATCH =
      Match(optional = false,
        Pattern(List(
          EveryPath(
            NodePattern(Some(aId), List(), None)(pos))
        ))(pos), List(), None)(pos)

    val pathExpression = PathExpression(NodePathStep(aId, NilPathStep))(pos)

    val WHERE =
      Where(
        equals(
          function("length", pathExpression),
          literalInt(1)
        )
      )(pos)

    val WITH =
      With(distinct = false,
        ReturnItems(includeExisting = false, Seq(
          AliasedReturnItem(aId, aId)(pos)
        ))(pos),
        Some(OrderBy(List(AscSortItem(pathExpression)(pos)))(pos)),
        None, None,
        Some(WHERE)
      )(pos)

    val RETURN =
      Return(distinct = false,
        ReturnItems(includeExisting = false, List(
          AliasedReturnItem(aId, aId)(pos)
        ))(pos),
        None, None, None
      )(pos)

    val expected: Query = Query(None, SingleQuery(List(MATCH, WITH, RETURN))(pos))(pos)

    rewritten should equal(expected)
  }
}
