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
package org.neo4j.cypher.internal.frontend

import org.neo4j.cypher.internal.ast
import org.neo4j.cypher.internal.ast.Statement
import org.neo4j.cypher.internal.ast.semantics.SemanticCheckResult
import org.neo4j.cypher.internal.ast.semantics.SemanticErrorDef
import org.neo4j.cypher.internal.ast.semantics.SemanticFeature
import org.neo4j.cypher.internal.ast.semantics.SemanticState
import org.neo4j.cypher.internal.frontend.helpers.TestContext
import org.neo4j.cypher.internal.frontend.helpers.TestState
import org.neo4j.cypher.internal.frontend.phases.PreparatoryRewriting
import org.neo4j.cypher.internal.parser
import org.neo4j.cypher.internal.parser.ParserTest
import org.neo4j.cypher.internal.rewriting.Deprecations
import org.parboiled.scala.Rule1

class MultipleGraphClauseSemanticCheckingTest
  extends ParserTest[ast.Statement, SemanticCheckResult] with parser.Statement {

  // INFO: Use result.dumpAndExit to debug these tests

  implicit val parser: Rule1[Statement] = Statement

  test("FROM requires feature") {
    parsingWith("""FROM g RETURN 1""", multiGraph)
      .shouldVerify(_.errorMessages.shouldEqual(Set("The `FROM GRAPH` clause is not available in this implementation of Cypher due to lack of support for FROM graph selector.")))
    parsingWith("""FROM g RETURN 1""", defaultFeatures)
      .shouldVerify(_.errorMessages.shouldEqual(Set()))
  }

  test("USE requires feature") {
    parsingWith("""USE g RETURN 1""", multiGraph)
      .shouldVerify(_.errorMessages.shouldEqual(Set("The `USE GRAPH` clause is not available in this implementation of Cypher due to lack of support for USE graph selector.")))
    parsingWith("""USE g RETURN 1""", useGraphSelector)
      .shouldVerify(_.errorMessages.shouldEqual(Set()))
  }

  // TODO: Need to reset semantic state after CONSTRUCT
  ignore("checks that all SET variables are in scope") {
    parsing(
      """|CONSTRUCT
         |  CREATE (a)
         |CONSTRUCT
         |  SET a :Label
         |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errorMessages should equal(Set("Variable `a` not defined"))
    }
  }

  test("from parameterised views") {
    parsing(
      """FROM GRAPH foo.bar(myView(grok.baz), a)
        |MATCH (a:A)
        |CONSTRUCT
        |  CREATE (a)-[:T]->(:B)
        |RETURN GRAPH""".stripMargin) shouldVerify { result => SemanticCheckResult

      result.errors shouldBe empty
    }
  }

  test("should allow parameterised from in normal query") {
    parsing(
      """FROM GRAPH $parameter
        |MATCH (a:A)
        |CONSTRUCT
        |  CREATE (a)-[:T]->(:B)
        |RETURN GRAPH""".stripMargin) shouldVerify { result => SemanticCheckResult

      result.errorMessages shouldBe empty
    }
  }

  test("parameterised views must use unique variables") {
    parsing(
      """CATALOG CREATE QUERY foo.bar($graph1, $graph1) {
        |  RETURN GRAPH
        |}""".stripMargin) shouldVerify { result => SemanticCheckResult

      result.errorMessages should equal(Set("Variable `$graph1` already declared"))
    }
  }

  test("parameterised views should allow normal variables with same name") {
    parsing(
      """CATALOG CREATE QUERY foo.bar($graph1) {
        |  FROM $graph1
        |  MATCH (graph1:A)
        |  RETURN GRAPH
        |}""".stripMargin) shouldVerify { result => SemanticCheckResult

      result.errorMessages shouldBe empty
    }
  }

  test("create parameterised views") {
    parsing(
      """CATALOG CREATE QUERY foo.bar($graph1, $graph2) {
        |  FROM $graph1
        |  MATCH (a:A)
        |  FROM $graph2
        |  MATCH (b:B)
        |  CONSTRUCT
        |    CREATE (a)-[:T]->(b)
        |  RETURN GRAPH
        |}""".stripMargin) shouldVerify { result => SemanticCheckResult

      result.errors shouldBe empty
    }
  }

  test("create empty parameterised views") {
    parsing(
      """CATALOG CREATE VIEW foo.bar() {
        |  FROM foo.bar
        |  MATCH (b:B)
        |  CONSTRUCT
        |    CREATE (a)-[:T]->(b)
        |  RETURN GRAPH
        |}""".stripMargin) shouldVerify { result => SemanticCheckResult

      result.errors shouldBe empty
    }
  }

  test("create empty parameterised views without parentheses") {
    parsing(
      """CATALOG CREATE VIEW foo.bar {
        |  FROM foo.bar()
        |  MATCH (b:B)
        |  CONSTRUCT
        |    CREATE (a)-[:T]->(b)
        |  RETURN GRAPH
        |}""".stripMargin) shouldVerify { result => SemanticCheckResult

      result.errors shouldBe empty
    }
  }

  test("allows both versions of FROM") {
    parsing(
      """FROM foo.bar
        |MATCH (a:Swedish)
        |CONSTRUCT
        |   CREATE (b COPY OF A:Programmer)
        |FROM GRAPH bar.foo
        |MATCH (a:Foo)
        |RETURN a.name""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errors shouldBe empty
    }
  }

  test("does not allow RETURN GRAPH in middle of query") {
    parsing(
      """MATCH (a:Swedish)
        |CONSTRUCT
        |   CREATE (b COPY OF A:Programmer)
        |RETURN GRAPH
        |MATCH (a:Foo)
        |RETURN a.name""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errorMessages should equal(Set("RETURN GRAPH can only be used at the end of the query"))
    }
  }

  test("allows COPY OF Node") {
    parsing(
      """MATCH (a:Swedish)
        |CONSTRUCT
        |   CREATE (b COPY OF A:Programmer)
        |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errors shouldBe empty
    }
  }

  test("allows COPY OF Node without new var") {
    parsing(
      """MATCH (a:Swedish)
        |CONSTRUCT
        |   CREATE (COPY OF A:Programmer)
        |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errors shouldBe empty
    }
  }

  test("fail on copy of with incompatible node type") {
    parsing(
      """MATCH ()-[r]->()
        |CONSTRUCT
        |   CREATE (b COPY OF r:Programmer)
        |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errorMessages should equal(Set("Type mismatch: r defined with conflicting type Relationship (expected Node)"))
    }
  }

  test("fail on copy of with incompatible rel type") {
    parsing(
      """MATCH (a)
        |CONSTRUCT
        |   CREATE ()-[r COPY OF a]->()
        |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errorMessages should equal(Set("Type mismatch: a defined with conflicting type Node (expected Relationship)"))
    }
  }

  test("fail on copy of with incompatible rel type chained") {
    parsing(
      """MATCH (a)
        |CONSTRUCT
        |   CREATE ()-[r COPY OF a]->()-[r2:REL]->()
        |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errorMessages should equal(Set("Type mismatch: a defined with conflicting type Node (expected Relationship)"))
    }
  }

  test("Do not require type for cloned relationships") {
    parsing(
      """MATCH (a)-[r]-(b)
        |CONSTRUCT
        |   CLONE a, r, b
        |   CREATE (a)-[r]->(b)
        |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errors shouldBe empty
    }
  }

  test("Do not require type for implicitly cloned relationships") {
    parsing(
      """MATCH (a)-[r]-(b)
        |CONSTRUCT
        |  CREATE (a)-[r]->(b)
        |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errors shouldBe empty
    }
  }

  test("Do not require type for aliased cloned relationships") {
    parsing(
      """MATCH (a)-[r]-(b)
        |CONSTRUCT
        |  CLONE r as newR
        |  CREATE (a)-[newR]->(b)
        |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errors shouldBe empty
    }
  }

  test("Require type for new relationships") {
    parsing(
      """MATCH (a), (b)
        |CONSTRUCT
        |   CLONE a, b
        |   CREATE (a)-[r]->(b)
        |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errorMessages should equal(
        Set("Exactly one relationship type must be specified for CREATE. Did you forget to prefix your relationship type with a ':'?")
      )
    }
  }

  test("Require type for new relationships chained") {
    parsing(
      """MATCH (a), (b)
        |CONSTRUCT
        |   CLONE a, b
        |   CREATE (a)-[r]->(b)-[r2:REL]->(c)
        |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errorMessages should equal(
        Set("Exactly one relationship type must be specified for CREATE. Did you forget to prefix your relationship type with a ':'?")
      )
    }
  }

  test("Allow multiple usages of a newly created nodes for connections") {
    parsing(
      """MATCH (a), (b)
        |CONSTRUCT
        |   CREATE (a2)
        |   CREATE (a2)-[r1:REL]->(b)
        |   CREATE (a2)-[r2:REL]->(a)
        |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errors shouldBe empty
    }
  }

  // TODO: Fix scoping of registered variables
  ignore("Do not allow multiple usages of a newly created node") {
    parsing(
      """MATCH (a), (b)
        |CONSTRUCT
        |   CREATE (a2)
        |   CREATE (a2)
        |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errorMessages should equal(
        Set("Variable `a2` already declared")
      )
    }
  }

  test("Do not allow multiple usages of a newly created relationship in NEW") {
    parsing(
      """MATCH (a)-[r]->(b)
        |CONSTRUCT
        |   CLONE a, b
        |   CREATE (a)-[r2:REL]->(b)
        |   CREATE (b)-[r2:REL]->(a)
        |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errorMessages should equal(
        Set("Relationship `r2` can only be declared once")
      )
    }
  }

  test("Do not allow to specify conflicting base nodes for new nodes") {
    parsing(
      """MATCH (a),(b)
        |CONSTRUCT
        |   CREATE (a2 COPY OF a)
        |   CREATE (a2 COPY OF b)-[r:REL]->(a)
        |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errorMessages should equal(
        Set("Node a2 cannot inherit from multiple bases a, b")
      )
    }
  }

  test("Do not allow manipulating labels of cloned aliased nodes") {
    parsing(
      """MATCH (a)
        |CONSTRUCT
        |   CLONE a as newA
        |   CREATE (newA:FOO)
        |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errorMessages should equal(
        Set("Modification of a cloned node is not allowed. Use COPY OF to manipulate the node")
      )
    }
  }

  test("Do not allow manipulating labels of cloned nodes") {
    parsing(
      """MATCH (a)
        |CONSTRUCT
        |   CLONE a
        |   CREATE (a:FOO)
        |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errorMessages should equal(
        Set("Modification of a cloned node is not allowed. Use COPY OF to manipulate the node")
      )
    }
  }

  test("Do not allow manipulating labels of implicitly cloned nodes") {
    parsing(
      """MATCH (a)
        |CONSTRUCT
        |   CREATE (a:FOO)
        |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errorMessages should equal(
        Set("Modification of a cloned node is not allowed. Use COPY OF to manipulate the node")
      )
    }
  }

  test("Do not allow manipulating properties of cloned aliased nodes") {
    parsing(
      """MATCH (a)
        |CONSTRUCT
        |   CLONE a as newA
        |   CREATE (newA {foo: "bar"})
        |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errorMessages should equal(
        Set("Modification of a cloned node is not allowed. Use COPY OF to manipulate the node")
      )
    }
  }

  test("Do not allow manipulating properties of cloned nodes") {
    parsing(
      """MATCH (a)
        |CONSTRUCT
        |   CLONE a
        |   CREATE (a {foo: "bar"})
        |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errorMessages should equal(
        Set("Modification of a cloned node is not allowed. Use COPY OF to manipulate the node")
      )
    }
  }

  test("Do not allow manipulating properties of implicitly cloned nodes") {
    parsing(
      """MATCH (a)
        |CONSTRUCT
        |   CREATE (a {foo: "bar"})
        |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errorMessages should equal(
        Set("Modification of a cloned node is not allowed. Use COPY OF to manipulate the node")
      )
    }
  }

  test("Do not allow manipulating types of cloned aliased relationships") {
    parsing(
      """MATCH (a)-[r]->(b)
        |CONSTRUCT
        |   CLONE a, r as newR, b
        |   CREATE (a)-[newR:FOO]->(b)
        |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errorMessages should equal(
        Set("Modification of a cloned relationship is not allowed. Use COPY OF to manipulate the relationship")
      )
    }
  }

  test("Do not allow manipulating types of cloned relationships") {
    parsing(
      """MATCH (a)-[r]->(b)
        |CONSTRUCT
        |   CLONE a, r, b
        |   CREATE (a)-[r:FOO]->(b)
        |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errorMessages should equal(
        Set("Modification of a cloned relationship is not allowed. Use COPY OF to manipulate the relationship")
      )
    }
  }

  test("Do not allow manipulating types of implicitly cloned relationships") {
    parsing(
      """MATCH (a)-[r]->(b)
        |CONSTRUCT
        |   CREATE (a)-[r:FOO]->(b)
        |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errorMessages should equal(
        Set("Modification of a cloned relationship is not allowed. Use COPY OF to manipulate the relationship")
      )
    }
  }

  test("Do not allow manipulating properties of cloned aliased relationships") {
    parsing(
      """MATCH (a)-[r]->(b)
        |CONSTRUCT
        |   CLONE a, r as newR, b
        |   CREATE (a)-[newR {foo: "bar"}]->(b)
        |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errorMessages should equal(
        Set("Modification of a cloned relationship is not allowed. Use COPY OF to manipulate the relationship")
      )
    }
  }

  test("Do not allow manipulating properties of cloned relationships") {
    parsing(
      """MATCH (a)-[r]->(b)
        |CONSTRUCT
        |   CLONE a, r, b
        |   CREATE (a)-[r {foo: "bar"}]->(b)
        |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errorMessages should equal(
        Set("Modification of a cloned relationship is not allowed. Use COPY OF to manipulate the relationship")
      )
    }
  }

  test("Do not allow manipulating properties of implicitly cloned relationships") {
    parsing(
      """MATCH (a)-[r]->(b)
        |CONSTRUCT
        |   CREATE (a)-[r {foo: "bar"}]->(b)
        |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errorMessages should equal(
        Set("Modification of a cloned relationship is not allowed. Use COPY OF to manipulate the relationship")
      )
    }
  }

  test("allow using set clauses instead of property patterns") {
    parsing(
      """|MATCH (a)
         |CONSTRUCT
         |  CREATE (a)
         |  SET a.prop = 10
         |  SET a.foo = 'hello'
         |RETURN GRAPH""".stripMargin) shouldVerify { result: SemanticCheckResult =>

      result.errors shouldBe empty
    }
  }

  test("Allow single identifier in FROM/USE") {
    parsingWith("FROM x RETURN 1", fromGraphSelector)
      .shouldVerify(_.errors.shouldEqual(Seq()))
    parsingWith("USE x RETURN 1", useGraphSelector)
      .shouldVerify(_.errors.shouldEqual(Seq()))
  }

  test("Allow qualified identifier in FROM/USE") {
    parsingWith("FROM x.y.z RETURN 1", fromGraphSelector)
      .shouldVerify(_.errors.shouldEqual(Seq()))
    parsingWith("USE x.y.z RETURN 1", useGraphSelector)
      .shouldVerify(_.errors.shouldEqual(Seq()))
  }

  test("Allow view invocation in FROM/USE") {
    parsingWith("FROM v(g, w(k)) RETURN 1", fromGraphSelector)
      .shouldVerify(_.errors.shouldEqual(Seq()))
    parsingWith("USE v(g, w(k)) RETURN 1", useGraphSelector)
      .shouldVerify(_.errors.shouldEqual(Seq()))
  }

  test("Allow qualified view invocation in FROM/USE") {
    parsingWith("FROM a.b.v(g, x.g, x.v(k)) RETURN 1", fromGraphSelector)
      .shouldVerify(_.errors.shouldEqual(Seq()))
    parsingWith("USE a.b.v(g, x.g, x.v(k)) RETURN 1", useGraphSelector)
      .shouldVerify(_.errors.shouldEqual(Seq()))
  }

  test("Do not allow arbitrary expressions in FROM/USE") {
    parsingWith("FROM 1 RETURN 1", fromGraphSelector)
      .shouldVerify(_.errorMessages.shouldEqual(Set("Invalid graph reference")))
    parsingWith("USE 1 RETURN 1", useGraphSelector)
      .shouldVerify(_.errorMessages.shouldEqual(Set("Invalid graph reference")))

    parsingWith("FROM 'a' RETURN 1", fromGraphSelector)
      .shouldVerify(_.errorMessages.shouldEqual(Set("Invalid graph reference")))
    parsingWith("USE 'a' RETURN 1", useGraphSelector)
      .shouldVerify(_.errorMessages.shouldEqual(Set("Invalid graph reference")))

    parsingWith("FROM [x] RETURN 1", fromGraphSelector)
      .shouldVerify(_.errorMessages.shouldEqual(Set("Invalid graph reference")))
    parsingWith("USE [x] RETURN 1", useGraphSelector)
      .shouldVerify(_.errorMessages.shouldEqual(Set("Invalid graph reference")))

    parsingWith("FROM 1 + 2 RETURN 1", fromGraphSelector)
      .shouldVerify(_.errorMessages.shouldEqual(Set("Invalid graph reference")))
    parsingWith("USE 1 + 2 RETURN 1", useGraphSelector)
      .shouldVerify(_.errorMessages.shouldEqual(Set("Invalid graph reference")))
  }

  test("Disallow expressions in view invocations") {
    parsingWith("FROM a.b.v(1, 1+2, 'x') RETURN 1", fromGraphSelector)
      .shouldVerify(_.errorMessages.shouldEqual(Set("Invalid graph reference")))
    parsingWith("USE a.b.v(1, 1+2, 'x') RETURN 1", useGraphSelector)
      .shouldVerify(_.errorMessages.shouldEqual(Set("Invalid graph reference")))
  }

  test("Allow expressions in view invocations (with feature flag)") {
    parsingWith("WITH 1 AS x FROM v(2, 'x', x, x+3) RETURN 1", fromGraphSelector ++ expressionsInViews)
      .shouldVerify(_.errors.shouldEqual(Seq()))
    parsingWith("WITH 1 AS x USE v(2, 'x', x, x+3) RETURN 1", useGraphSelector ++ expressionsInViews)
      .shouldVerify(_.errors.shouldEqual(Seq()))
  }

  test("Expressions in view invocations are checked (with feature flag)") {
    parsingWith("WITH 1 AS x FROM v(2, 'x', y, x+3) RETURN 1", fromGraphSelector ++ expressionsInViews)
      .shouldVerify(_.errorMessages.shouldEqual(Set("Variable `y` not defined")))
    parsingWith("WITH 1 AS x USE v(2, 'x', y, x+3) RETURN 1", useGraphSelector ++ expressionsInViews)
      .shouldVerify(_.errorMessages.shouldEqual(Set("Variable `y` not defined")))
  }

  override type Extra = Seq[SemanticFeature]

  private val multiGraph = Seq(
    SemanticFeature.MultipleGraphs,
    SemanticFeature.WithInitialQuerySignature,
  )

  private val fromGraphSelector =
    multiGraph :+ SemanticFeature.FromGraphSelector

  private val useGraphSelector =
    multiGraph :+ SemanticFeature.UseGraphSelector

  private val defaultFeatures =
    fromGraphSelector

  private val expressionsInViews = Seq(
    SemanticFeature.ExpressionsInViewInvocations
  )



  override def convert(astNode: ast.Statement): SemanticCheckResult =
    convert(astNode, defaultFeatures)

  override def convert(astNode: Statement, features: Seq[SemanticFeature]): SemanticCheckResult = {
    val rewritten = PreparatoryRewriting(Deprecations.V1).transform(TestState(Some(astNode)), TestContext()).statement()
    val initialState =
      SemanticState.clean.withFeatures(features: _*)
    rewritten.semanticCheck(initialState)
  }

  implicit final class RichSemanticCheckResult(val result: SemanticCheckResult) {
    def state: SemanticState = result.state

    def errors: Seq[SemanticErrorDef] = result.errors

    def errorMessages: Set[String] = errors.map(_.msg).toSet
  }
}
