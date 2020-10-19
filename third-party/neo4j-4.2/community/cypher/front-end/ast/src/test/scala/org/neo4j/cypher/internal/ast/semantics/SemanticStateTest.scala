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
package org.neo4j.cypher.internal.ast.semantics

import org.neo4j.cypher.internal.expressions.DummyExpression
import org.neo4j.cypher.internal.expressions.Property
import org.neo4j.cypher.internal.expressions.PropertyKeyName
import org.neo4j.cypher.internal.expressions.Variable
import org.neo4j.cypher.internal.util.DummyPosition
import org.neo4j.cypher.internal.util.symbols.CTAny
import org.neo4j.cypher.internal.util.symbols.CTInteger
import org.neo4j.cypher.internal.util.symbols.CTMap
import org.neo4j.cypher.internal.util.symbols.CTNode
import org.neo4j.cypher.internal.util.symbols.CTNumber
import org.neo4j.cypher.internal.util.symbols.CTRelationship
import org.neo4j.cypher.internal.util.symbols.CTString
import org.neo4j.cypher.internal.util.symbols.TypeSpec
import org.neo4j.cypher.internal.util.test_helpers.CypherFunSuite

class SemanticStateTest extends CypherFunSuite {

  test("should declare variable once") {
    val variable1 = Variable("foo")(DummyPosition(0))
    val variable2 = Variable("foo")(DummyPosition(3))
    val state = SemanticState.clean.declareVariable(variable1, CTNode).right.get

    state.declareVariable(variable2, CTNode) match {
      case Right(_) => fail("Expected an error from second declaration")
      case Left(error) =>
        error.position should equal(variable2.position)
    }
  }

  test("should collect all variables when implicitly declared") {
    val variable1 = Variable("foo")(DummyPosition(0))
    val variable2 = Variable("foo")(DummyPosition(2))
    val variable3 = Variable("foo")(DummyPosition(3))

    SemanticState.clean.implicitVariable(variable1, CTNode) chain
      ((_: SemanticState).implicitVariable(variable2, CTNode)) chain
      ((_: SemanticState).implicitVariable(variable3, CTNode)) match {
      case Left(_) => fail("Expected success")
      case Right(state) =>
        val positions = state.currentScope.localSymbol("foo").map(_.positions).get
        positions should equal(Set(variable1.position, variable2.position, variable3.position))
    }
  }

  test("should constrain types for consecutive implicit variable declarations") {
    val variable1 = Variable("foo")(DummyPosition(0))
    val variable2 = Variable("foo")(DummyPosition(3))

    SemanticState.clean.implicitVariable(variable1, CTNode | CTRelationship) chain
      ((_: SemanticState).implicitVariable(variable2, CTNode)) match {
      case Left(_) => fail("Expected success")
      case Right(state) =>
        val types = state.symbolTypes("foo")
        types should equal(CTNode: TypeSpec)
    }

    SemanticState.clean.implicitVariable(variable1, CTRelationship) chain
      ((_: SemanticState).implicitVariable(variable2, CTNode | CTRelationship)) match {
      case Left(_) => fail("Expected success")
      case Right(state) =>
        val types = state.symbolTypes("foo")
        types should equal(CTRelationship: TypeSpec)
    }

    SemanticState.clean.implicitVariable(variable1, CTNode | CTRelationship) chain
      ((_: SemanticState).implicitVariable(variable2, CTAny.covariant)) match {
      case Left(_) => fail("Expected success")
      case Right(state) =>
        val types = state.symbolTypes("foo")
        types should equal(CTNode | CTRelationship)
    }

    SemanticState.clean.implicitVariable(variable1, CTNode) chain
      ((_: SemanticState).implicitVariable(variable2, CTMap.covariant)) match {
      case Left(_) => fail("Expected success")
      case Right(state) =>
        val types = state.symbolTypes("foo")
        types should equal(CTNode: TypeSpec)
    }
  }

  test("should fail if no possible types remain after implicit variable declaration") {
    SemanticState.clean.implicitVariable(Variable("foo")(DummyPosition(0)), CTMap) chain
      ((_: SemanticState).implicitVariable(Variable("foo")(DummyPosition(3)), CTNode)) match {
      case Right(_) => fail("Expected an error")
      case Left(error) =>
        error.position should equal(DummyPosition(3))
        error.msg should equal("Type mismatch: foo defined with conflicting type Map (expected Node)")
    }

    SemanticState.clean.implicitVariable(Variable("foo")(DummyPosition(0)), CTNode | CTRelationship) chain
      ((_: SemanticState).implicitVariable(Variable("foo")(DummyPosition(3)), CTNode | CTInteger)) chain
      ((_: SemanticState).implicitVariable(Variable("foo")(DummyPosition(9)), CTInteger | CTRelationship)) match {
      case Right(_) => fail("Expected an error")
      case Left(error) =>
        error.position should equal(DummyPosition(9))
        error.msg should equal("Type mismatch: foo defined with conflicting type Node (expected Integer or Relationship)")
    }
  }

  test("should record type for expression when specifying type") {
    val expression = DummyExpression(CTInteger | CTString)
    val state = SemanticState.clean.specifyType(expression, expression.possibleTypes).right.get
    state.expressionType(expression).specified should equal(expression.possibleTypes)
    state.expressionType(expression).actual should equal(expression.possibleTypes)
  }

  test("should expect type for expression") {
    val expression = DummyExpression(CTInteger | CTString | CTMap)
    val state = SemanticState.clean.specifyType(expression, expression.possibleTypes).right.get

    state.expectType(expression, CTNumber.covariant) match {
      case (s, typ) =>
        typ should equal(CTInteger: TypeSpec)
        s.expressionType(expression).actual should equal(typ)
    }

    state.expectType(expression, CTNode.covariant | CTNumber.covariant) match {
      case (s, typ) =>
        typ should equal(CTInteger: TypeSpec)
        s.expressionType(expression).actual should equal(typ)
    }
  }

  test("should find symbol in parent") {
    val s1 = SemanticState.clean.declareVariable(Variable("foo")(DummyPosition(0)), CTNode).right.get
    val s2 = s1.newChildScope
    s2.symbolTypes("foo") should equal(CTNode: TypeSpec)
  }

  test("should override symbol in parent") {
    val s1 = SemanticState.clean.declareVariable(Variable("foo")(DummyPosition(0)), CTNode).right.get
    val s2 = s1.newChildScope.declareVariable(Variable("foo")(DummyPosition(0)), CTString).right.get

    s2.symbolTypes("foo") should equal(CTString: TypeSpec)
  }

  test("should extend symbol in parent") {
    val s1 = SemanticState.clean.declareVariable(Variable("foo")(DummyPosition(0)), CTNode).right.get
    val s2 = s1.newChildScope.implicitVariable(Variable("foo")(DummyPosition(0)), CTAny.covariant).right.get
    s2.symbolTypes("foo") should equal(CTNode: TypeSpec)
  }

  test("should return types of variable") {
    val variable = Variable("foo")(DummyPosition(0))
    val s1 = SemanticState.clean.declareVariable(variable, CTNode).right.get
    s1.expressionType(variable).actual should equal(CTNode: TypeSpec)
  }

  test("should return types of variable at later expression") {
    val variable1 = Variable("foo")(DummyPosition(0))
    val variable2 = Variable("foo")(DummyPosition(3))
    val s1 = SemanticState.clean.declareVariable(variable1, CTNode).right.get
    val s2 = s1.implicitVariable(variable2, CTNode).right.get
    s2.expressionType(variable2).actual should equal(CTNode: TypeSpec)
  }

  test("should maintain separate TypeInfo for equivalent expressions") {
    val exp1 = Property(Variable("n")(DummyPosition(0)), PropertyKeyName("prop")(DummyPosition(3)))(DummyPosition(0))
    val exp2 = Property(Variable("n")(DummyPosition(6)), PropertyKeyName("prop")(DummyPosition(9)))(DummyPosition(6))
    val s1 = SemanticState.clean.specifyType(exp1, CTNode).right.get
    val s2 = s1.specifyType(exp2, CTRelationship).right.get

    s2.expressionType(exp1).specified should equal(CTNode: TypeSpec)
    s2.expressionType(exp2).specified should equal(CTRelationship: TypeSpec)

    val s3 = s2.expectType(exp1, CTMap)._1.expectType(exp2, CTAny)._1
    s3.expressionType(exp1).expected should equal(Some(CTMap: TypeSpec))
    s3.expressionType(exp2).expected should equal(Some(CTAny: TypeSpec))
  }

  test("should gracefully update a variable") {
    val s1 = SemanticState.clean.declareVariable(Variable("foo")(DummyPosition(0)), CTNode).right.get
    val s2: SemanticState = s1.newChildScope.declareVariable(Variable("foo")(DummyPosition(0)), CTRelationship).right.get
    s1.symbolTypes("foo") should equal(CTNode.invariant)
    s2.symbolTypes("foo") should equal(CTRelationship.invariant)
  }

  test("should be able to import scopes") {
    val s1 =
      SemanticState.clean
        .declareVariable(Variable("foo")(DummyPosition(0)), CTNode).right.get
        .declareVariable(Variable("bar")(DummyPosition(1)), CTNode).right.get


    val s2 =
      SemanticState.clean
        .declareVariable(Variable("foo")(DummyPosition(1)), CTNode).right.get
        .declareVariable(Variable("baz")(DummyPosition(4)), CTNode).right.get

    val actual = s1.importValuesFromScope(s2.scopeTree)
    val expected =
      SemanticState.clean
        .declareVariable(Variable("foo")(DummyPosition(1)), CTNode).right.get
        .declareVariable(Variable("bar")(DummyPosition(1)), CTNode).right.get
        .declareVariable(Variable("baz")(DummyPosition(4)), CTNode).right.get


    actual.scopeTree should equal(expected.scopeTree)
  }

  test("should be able to import scopes and honor excludes") {
    val s1 =
      SemanticState.clean
        .declareVariable(Variable("foo")(DummyPosition(0)), CTNode).right.get
        .declareVariable(Variable("bar")(DummyPosition(1)), CTNode).right.get


    val s2 =
      SemanticState.clean
        .declareVariable(Variable("foo")(DummyPosition(1)), CTNode).right.get
        .declareVariable(Variable("baz")(DummyPosition(4)), CTNode).right.get
        .declareVariable(Variable("frob")(DummyPosition(5)), CTNode).right.get

    val actual = s1.importValuesFromScope(s2.scopeTree, Set("foo", "frob"))
    val expected =
      SemanticState.clean
        .declareVariable(Variable("foo")(DummyPosition(0)), CTNode).right.get
        .declareVariable(Variable("bar")(DummyPosition(1)), CTNode).right.get
        .declareVariable(Variable("baz")(DummyPosition(4)), CTNode).right.get


    actual.scopeTree should equal(expected.scopeTree)
  }

  implicit class ChainableSemanticStateEither(either: Either[SemanticError, SemanticState]) {
    def chain(next: SemanticState => Either[SemanticError, SemanticState]): Either[SemanticError, SemanticState] = {
      either match {
        case Left(_)      => either
        case Right(state) => next(state)
      }
    }
  }
}
