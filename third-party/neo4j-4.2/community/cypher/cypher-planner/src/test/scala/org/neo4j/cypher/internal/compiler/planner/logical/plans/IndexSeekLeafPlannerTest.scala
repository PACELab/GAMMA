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
package org.neo4j.cypher.internal.compiler.planner.logical.plans

import org.neo4j.cypher.internal.ast.UsingIndexHint
import org.neo4j.cypher.internal.compiler.planner.BeLikeMatcher.beLike
import org.neo4j.cypher.internal.compiler.planner.LogicalPlanningTestSupport2
import org.neo4j.cypher.internal.compiler.planner.logical.steps.indexSeekLeafPlanner
import org.neo4j.cypher.internal.compiler.planner.logical.steps.mergeUniqueIndexSeekLeafPlanner
import org.neo4j.cypher.internal.expressions.AndedPropertyInequalities
import org.neo4j.cypher.internal.expressions.Expression
import org.neo4j.cypher.internal.expressions.LabelToken
import org.neo4j.cypher.internal.expressions.Property
import org.neo4j.cypher.internal.expressions.PropertyKeyName
import org.neo4j.cypher.internal.expressions.PropertyKeyToken
import org.neo4j.cypher.internal.ir.Predicate
import org.neo4j.cypher.internal.ir.QueryGraph
import org.neo4j.cypher.internal.ir.Selections
import org.neo4j.cypher.internal.ir.ordering.InterestingOrder
import org.neo4j.cypher.internal.logical.plans.AssertSameNode
import org.neo4j.cypher.internal.logical.plans.CanGetValue
import org.neo4j.cypher.internal.logical.plans.CompositeQueryExpression
import org.neo4j.cypher.internal.logical.plans.DoNotGetValue
import org.neo4j.cypher.internal.logical.plans.ExclusiveBound
import org.neo4j.cypher.internal.logical.plans.IndexOrderNone
import org.neo4j.cypher.internal.logical.plans.IndexedProperty
import org.neo4j.cypher.internal.logical.plans.InequalitySeekRangeWrapper
import org.neo4j.cypher.internal.logical.plans.LogicalPlan
import org.neo4j.cypher.internal.logical.plans.NodeIndexSeek
import org.neo4j.cypher.internal.logical.plans.NodeUniqueIndexSeek
import org.neo4j.cypher.internal.logical.plans.RangeLessThan
import org.neo4j.cypher.internal.logical.plans.RangeQueryExpression
import org.neo4j.cypher.internal.logical.plans.SingleQueryExpression
import org.neo4j.cypher.internal.util.LabelId
import org.neo4j.cypher.internal.util.NonEmptyList
import org.neo4j.cypher.internal.util.PropertyKeyId
import org.neo4j.cypher.internal.util.symbols.CTInteger
import org.neo4j.cypher.internal.util.symbols.CTNode
import org.neo4j.cypher.internal.util.symbols.CTString
import org.neo4j.cypher.internal.util.test_helpers.CypherFunSuite

class IndexSeekLeafPlannerTest extends CypherFunSuite with LogicalPlanningTestSupport2 {

  private val idName = "n"
  private val property = prop("n", "prop")
  private val lit42: Expression = literalInt(42)
  private val lit6: Expression = literalInt(6)

  private val inPredicate = in(property, listOf(lit42))
  private val lessThanPredicate = AndedPropertyInequalities(varFor("n"), property, NonEmptyList(lessThan(property, lit42)))

  private def hasLabel(l: String) = hasLabels("n", l)

  test("does not plan index seek when no index exist") {
    new given {
      addTypeToSemanticTable(lit42, CTInteger.invariant)
      addTypeToSemanticTable(lit42, CTInteger.invariant)
      qg = queryGraph(inPredicate, hasLabel("Awesome"))
    }.withLogicalPlanningContext { (cfg, ctx) =>
      // when
      val resultPlans = indexSeekLeafPlanner(cfg.qg, InterestingOrder.empty, ctx)

      // then
      resultPlans shouldBe empty
    }
  }

  test("does plan unique index seek when the index is unique") {
    new given {
      addTypeToSemanticTable(lit42, CTInteger.invariant)
      addTypeToSemanticTable(lit42, CTInteger.invariant)
      qg = queryGraph(inPredicate, hasLabel("Awesome"))
      uniqueIndexOn("Awesome", "prop")
    }.withLogicalPlanningContext { (cfg, ctx) =>
      // when
      val resultPlans = indexSeekLeafPlanner(cfg.qg, InterestingOrder.empty, ctx)

      // then
      resultPlans.head should beLike {
        case _: NodeUniqueIndexSeek => ()
      }
    }
  }

  test("index seek with values (equality predicate) when there is an index on the property") {
    new given {
      addTypeToSemanticTable(lit42, CTInteger.invariant)
      qg = queryGraph(inPredicate, hasLabel("Awesome"))

      indexOn("Awesome", "prop")
    }.withLogicalPlanningContext { (cfg, ctx) =>
      // when
      val resultPlans = indexSeekLeafPlanner(cfg.qg, InterestingOrder.empty, ctx)

      // then
      resultPlans should beLike {
        case Seq(NodeIndexSeek(`idName`, _, Seq(IndexedProperty(_, CanGetValue)), SingleQueryExpression(`lit42`), _, _)) => ()
      }
    }
  }

  test("index seeks when there is an index on the property and there are multiple predicates") {
    val prop1: Property = prop("n", "prop")
    val prop1Predicate1 = in(prop1, listOf(lit42))
    val prop1Predicate2 = AndedPropertyInequalities(varFor("n"), prop1, NonEmptyList(lessThan(prop1, lit6)))
    val prop1Predicate1Expr = SingleQueryExpression(lit42)
    val prop1Predicate2Expr = RangeQueryExpression(InequalitySeekRangeWrapper(RangeLessThan(NonEmptyList(ExclusiveBound(lit6))))(pos))

    new given {
      addTypeToSemanticTable(lit42, CTInteger.invariant)
      addTypeToSemanticTable(lit6, CTInteger.invariant)
      qg = queryGraph(prop1Predicate1, prop1Predicate2, hasLabel("Awesome"))

      indexOn("Awesome", "prop")
    }.withLogicalPlanningContext { (cfg, ctx) =>
      // when
      val resultPlans = indexSeekLeafPlanner(cfg.qg, InterestingOrder.empty, ctx).toSet

      // then
      val labelToken = LabelToken("Awesome", LabelId(0))
      val prop1Token = PropertyKeyToken("prop", PropertyKeyId(0))

      val expected = Set(
        NodeIndexSeek(idName, labelToken, Seq(IndexedProperty(prop1Token, CanGetValue)), prop1Predicate1Expr, Set(), IndexOrderNone),
        NodeIndexSeek(idName, labelToken, Seq(IndexedProperty(prop1Token, DoNotGetValue)), prop1Predicate2Expr, Set(), IndexOrderNone),
      )

      resultPlans shouldEqual expected
    }
  }

  test("index seeks when there is a composite index and there are multiple predicates") {

    val lit43: Expression = literalInt(43)
    val lit44: Expression = literalInt(44)
    val prop1: Property = prop("n", "prop")
    val prop2: Property = prop("n", "prop2")
    val prop1Predicate1 = in(prop1, listOf(lit42))
    val prop1Predicate2 = in(prop1, listOf(lit43))
    val prop1Predicate3 = in(prop1, listOf(lit44))
    val prop2Predicate1 = in(prop2, listOf(lit6))
    val prop2Predicate2 = AndedPropertyInequalities(varFor("n"), prop2, NonEmptyList(lessThan(prop2, lit6)))
    val prop1Predicate1Expr = SingleQueryExpression(lit42)
    val prop1Predicate2Expr = SingleQueryExpression(lit43)
    val prop1Predicate3Expr = SingleQueryExpression(lit44)
    val prop2Predicate1Expr = SingleQueryExpression(lit6)
    val prop2Predicate2Expr = RangeQueryExpression(InequalitySeekRangeWrapper(RangeLessThan(NonEmptyList(ExclusiveBound(lit6))))(pos))

    new given {
      addTypeToSemanticTable(lit42, CTInteger.invariant)
      addTypeToSemanticTable(lit6, CTInteger.invariant)

      qg = queryGraph(prop1Predicate1, prop1Predicate2, prop2Predicate1, prop2Predicate2, prop1Predicate3, hasLabel("Awesome"))

      indexOn("Awesome", "prop", "prop2")
    }.withLogicalPlanningContext { (cfg, ctx) =>
      // when
      val resultPlans = indexSeekLeafPlanner(cfg.qg, InterestingOrder.empty, ctx).toSet

      // then
      val labelToken = LabelToken("Awesome", LabelId(0))
      val prop1Token = PropertyKeyToken("prop", PropertyKeyId(0))
      val prop2Token = PropertyKeyToken("prop2", PropertyKeyId(1))

      val expected = Set(
        NodeIndexSeek(idName, labelToken, Seq(IndexedProperty(prop1Token, CanGetValue), IndexedProperty(prop2Token, CanGetValue)),   CompositeQueryExpression(Seq(prop1Predicate1Expr, prop2Predicate1Expr)), Set(), IndexOrderNone),
        NodeIndexSeek(idName, labelToken, Seq(IndexedProperty(prop1Token, CanGetValue), IndexedProperty(prop2Token, DoNotGetValue)), CompositeQueryExpression(Seq(prop1Predicate1Expr, prop2Predicate2Expr)), Set(), IndexOrderNone),
        NodeIndexSeek(idName, labelToken, Seq(IndexedProperty(prop1Token, CanGetValue), IndexedProperty(prop2Token, CanGetValue)),   CompositeQueryExpression(Seq(prop1Predicate2Expr, prop2Predicate1Expr)), Set(), IndexOrderNone),
        NodeIndexSeek(idName, labelToken, Seq(IndexedProperty(prop1Token, CanGetValue), IndexedProperty(prop2Token, DoNotGetValue)), CompositeQueryExpression(Seq(prop1Predicate2Expr, prop2Predicate2Expr)), Set(), IndexOrderNone),
        NodeIndexSeek(idName, labelToken, Seq(IndexedProperty(prop1Token, CanGetValue), IndexedProperty(prop2Token, CanGetValue)),   CompositeQueryExpression(Seq(prop1Predicate3Expr, prop2Predicate1Expr)), Set(), IndexOrderNone),
        NodeIndexSeek(idName, labelToken, Seq(IndexedProperty(prop1Token, CanGetValue), IndexedProperty(prop2Token, DoNotGetValue)), CompositeQueryExpression(Seq(prop1Predicate3Expr, prop2Predicate2Expr)), Set(), IndexOrderNone),
      )

      resultPlans shouldEqual expected
    }
  }

  test("index seeks when there are multiple composite indexes and there are multiple predicates") {

    val lit43: Expression = literalInt(43)
    val prop1: Property = prop("n", "prop")
    val prop2: Property = prop("n", "prop2")
    val prop1Predicate1 = in(prop1, listOf(lit42))
    val prop1Predicate2 = in(prop1, listOf(lit43))
    val prop2Predicate1 = in(prop2, listOf(lit6))
    val prop1Predicate1Expr = SingleQueryExpression(lit42)
    val prop1Predicate2Expr = SingleQueryExpression(lit43)
    val prop2Predicate1Expr = SingleQueryExpression(lit6)

    new given {
      addTypeToSemanticTable(lit42, CTInteger.invariant)
      addTypeToSemanticTable(lit6, CTInteger.invariant)

      qg = queryGraph(prop2Predicate1, prop1Predicate1, prop1Predicate2, hasLabel("Awesome"))

      indexOn("Awesome", "prop", "prop2")
      indexOn("Awesome", "prop2", "prop")
    }.withLogicalPlanningContext { (cfg, ctx) =>
      // when
      val resultPlans = indexSeekLeafPlanner(cfg.qg, InterestingOrder.empty, ctx).toSet

      // then
      val labelToken = LabelToken("Awesome", LabelId(0))
      val prop1Token = PropertyKeyToken("prop", PropertyKeyId(0))
      val prop2Token = PropertyKeyToken("prop2", PropertyKeyId(1))

      val expected = Set(
        NodeIndexSeek(idName, labelToken, Seq(IndexedProperty(prop1Token, CanGetValue), IndexedProperty(prop2Token, CanGetValue)),   CompositeQueryExpression(Seq(prop1Predicate1Expr, prop2Predicate1Expr)), Set(), IndexOrderNone),
        NodeIndexSeek(idName, labelToken, Seq(IndexedProperty(prop1Token, CanGetValue), IndexedProperty(prop2Token, CanGetValue)),   CompositeQueryExpression(Seq(prop1Predicate2Expr, prop2Predicate1Expr)), Set(), IndexOrderNone),
        NodeIndexSeek(idName, labelToken, Seq(IndexedProperty(prop2Token, CanGetValue), IndexedProperty(prop1Token, CanGetValue)),   CompositeQueryExpression(Seq(prop2Predicate1Expr, prop1Predicate1Expr)), Set(), IndexOrderNone),
        NodeIndexSeek(idName, labelToken, Seq(IndexedProperty(prop2Token, CanGetValue), IndexedProperty(prop1Token, CanGetValue)),   CompositeQueryExpression(Seq(prop2Predicate1Expr, prop1Predicate2Expr)), Set(), IndexOrderNone),
      )

      resultPlans shouldEqual expected
    }
  }

  test("index seeks when there is a composite index and there are multiple predicates that do not cover all properties") {

    val lit43: Expression = literalInt(43)
    val prop1: Property = prop("n", "prop")
    val prop2: Property = prop("n", "prop2")
    val prop1Predicate1 = in(prop1, listOf(lit42))
    val prop1Predicate2 = in(prop1, listOf(lit43))
    val prop2Predicate1 = in(prop2, listOf(lit6))
    val prop2Predicate2 = AndedPropertyInequalities(varFor("n"), prop2, NonEmptyList(lessThan(prop2, lit6)))

    new given {
      addTypeToSemanticTable(lit42, CTInteger.invariant)
      addTypeToSemanticTable(lit6, CTInteger.invariant)

      qg = queryGraph(prop1Predicate1, prop1Predicate2, prop2Predicate1, prop2Predicate2, hasLabel("Awesome"))

      indexOn("Awesome", "prop", "prop2", "prop3")
    }.withLogicalPlanningContext { (cfg, ctx) =>
      // when
      val resultPlans = indexSeekLeafPlanner(cfg.qg, InterestingOrder.empty, ctx).toSet

      // then
      val expected = Set()

      resultPlans shouldEqual expected
    }
  }

  test("index seek without values when there is an index on the property") {
    new given {
      addTypeToSemanticTable(lit42, CTInteger.invariant)
      qg = queryGraph(lessThanPredicate, hasLabel("Awesome"))

      indexOn("Awesome", "prop")
    }.withLogicalPlanningContext { (cfg, ctx) =>
      // when
      val resultPlans = indexSeekLeafPlanner(cfg.qg, InterestingOrder.empty, ctx)

      // then
      resultPlans should beLike {
        case Seq(NodeIndexSeek(`idName`, _, Seq(IndexedProperty(_, DoNotGetValue)), _, _, _)) => ()
      }
    }
  }

  test("index seek with values (from index) when there is an index on the property") {
    new given {
      addTypeToSemanticTable(lit42, CTInteger.invariant)
      qg = queryGraph(lessThanPredicate, hasLabel("Awesome"))

      indexOn("Awesome", "prop").providesValues()
    }.withLogicalPlanningContext { (cfg, ctx) =>
      // when
      val resultPlans = indexSeekLeafPlanner(cfg.qg, InterestingOrder.empty, ctx)

      // then
      resultPlans should beLike {
        case Seq(NodeIndexSeek(`idName`, _, Seq(IndexedProperty(_, CanGetValue)), _, _, _)) => ()
      }
    }
  }

  test("index seek with values (equality predicate) when there is a composite index on two properties") {
    new given {
      addTypeToSemanticTable(lit6, CTInteger.invariant)
      addTypeToSemanticTable(lit42, CTInteger.invariant)
      private val inPredicate2 = in(prop("n", "prop2"), listOf(lit6))
      qg = queryGraph(inPredicate, inPredicate2, hasLabel("Awesome"))

      indexOn("Awesome", "prop", "prop2")
    }.withLogicalPlanningContext { (cfg, ctx) =>
      // when
      val resultPlans = indexSeekLeafPlanner(cfg.qg, InterestingOrder.empty, ctx)

      // then
      resultPlans should beLike {
        case Seq(NodeIndexSeek(`idName`, LabelToken("Awesome", _),
        Seq(IndexedProperty(PropertyKeyToken("prop", _), CanGetValue), IndexedProperty(PropertyKeyToken("prop2", _), CanGetValue)),
        CompositeQueryExpression(Seq(SingleQueryExpression(`lit42`), SingleQueryExpression(`lit6`))), _, _)) => ()
      }
    }
  }

  test("index seek with values (equality predicate) when there is a composite index on two properties in the presence of other nodes, labels and properties") {
    new given {
      addTypeToSemanticTable(lit6, CTInteger.invariant)
      addTypeToSemanticTable(lit42, CTInteger.invariant)
      private val litFoo = literalString("foo")
      addTypeToSemanticTable(litFoo, CTString.invariant)

      // MATCH (n:Awesome:Sauce), (m:Awesome)
      // WHERE n.prop = 42 AND n.prop2 = 6 AND n.prop3 = "foo" AND m.prop = "foo"
      qg = queryGraph(
        // node 'n'
        hasLabel("Awesome"),
        hasLabel("Sauce"),
        in(prop("n", "prop"), listOf(lit42)),
        in(prop("n", "prop2"), listOf(lit6)),
        in(prop("n", "prop3"), listOf(litFoo)),
        // node 'm'
        hasLabels("m", "Awesome"),
        in(prop("m", "prop"), listOf(litFoo))
      )

      // CREATE INDEX FOR (n:Awesome) ON (n.prop, n.prop2)
      indexOn("Awesome", "prop", "prop2")

    }.withLogicalPlanningContext { (cfg, ctx) =>

      // when
      val resultPlans = indexSeekLeafPlanner(cfg.qg, InterestingOrder.empty, ctx)

      // then
      resultPlans should beLike {
        case Seq(NodeIndexSeek(`idName`, LabelToken("Awesome", _),
        Seq(IndexedProperty(PropertyKeyToken("prop", _), CanGetValue), IndexedProperty(PropertyKeyToken("prop2", _), CanGetValue)),
        CompositeQueryExpression(Seq(SingleQueryExpression(`lit42`), SingleQueryExpression(`lit6`))), _, _)) => ()
      }
    }
  }

  test("index seek with values (equality predicate) when there is a composite index on many properties") {
    val propertyNames = (0 to 10).map(n => s"prop$n")
    val properties = propertyNames.map(n => prop("n", n))
    val values = (0 to 10).map(n => literalInt(n * 10 + 2))
    val predicates = properties.zip(values).map { pair =>
      val predicate = in(pair._1, listOf(pair._2))
      Predicate(Set(idName), predicate)
    }

    new given {
      addTypeToSemanticTable(lit42, CTInteger.invariant)
      values.foreach(addTypeToSemanticTable(_, CTInteger.invariant))
      qg = QueryGraph(
        selections = Selections(predicates.toSet + Predicate(Set(idName), hasLabel("Awesome"))),
        patternNodes = Set(idName)
      )

      indexOn("Awesome", propertyNames:_*)
    }.withLogicalPlanningContext { (cfg, ctx) =>
      // when
      val resultPlans = indexSeekLeafPlanner(cfg.qg, InterestingOrder.empty, ctx)

      // then
      resultPlans should beLike {
        case Seq(NodeIndexSeek(
        `idName`,
        LabelToken("Awesome", _),
        props@Seq(_*),
        CompositeQueryExpression(vals@Seq(_*)),
        _,
        _))
          if assertPropsAndValuesMatch(propertyNames, values, props, vals.flatMap(_.expressions)) => ()
      }
    }
  }

  private def assertPropsAndValuesMatch(expectedProps: Seq[String], expectedVals: Seq[Expression], foundProps: Seq[IndexedProperty], foundVals: Seq[Expression]) = {
    val expected = expectedProps.zip(expectedVals).toMap
    val found = foundProps.map(_.propertyKeyToken.name).zip(foundVals).toMap
    found.equals(expected)
  }

  test("plans index seeks when variable exists as an argument") {
    new given {
      addTypeToSemanticTable(lit42, CTInteger.invariant)
      // GIVEN 42 as x MATCH a WHERE a.prop IN [x]
      val x: Expression = varFor("x")
      qg = queryGraph(in(property, listOf(x)), hasLabel("Awesome")).addArgumentIds(Seq("x"))

      addTypeToSemanticTable(x, CTNode.invariant)
      indexOn("Awesome", "prop")
    }.withLogicalPlanningContext { (cfg, ctx) =>
      // when
      val x = cfg.x
      val resultPlans = indexSeekLeafPlanner(cfg.qg, InterestingOrder.empty, ctx)

      // then
      resultPlans should beLike {
        case Seq(NodeIndexSeek(`idName`, _, _, SingleQueryExpression(`x`), _, _)) => ()
      }
    }
  }

  test("does not plan an index seek when the RHS expression does not have its dependencies in scope") {
    new given {
      addTypeToSemanticTable(lit42, CTInteger.invariant)
      // MATCH a, x WHERE a.prop IN [x]
      qg = queryGraph(in(property, listOf(varFor("x"))), hasLabel("Awesome"))

      indexOn("Awesome", "prop")
    }.withLogicalPlanningContext { (cfg, ctx) =>
      // when
      val resultPlans = indexSeekLeafPlanner(cfg.qg, InterestingOrder.empty, ctx)

      // then
      resultPlans shouldBe empty
    }
  }

  test("unique index seek with values (equality predicate) when there is an unique index on the property") {
    new given {
      addTypeToSemanticTable(lit42, CTInteger.invariant)
      qg = queryGraph(inPredicate, hasLabel("Awesome"))

      uniqueIndexOn("Awesome", "prop")
    }.withLogicalPlanningContext { (cfg, ctx) =>
      // when
      val resultPlans = indexSeekLeafPlanner(cfg.qg, InterestingOrder.empty, ctx)

      // then
      resultPlans should beLike {
        case Seq(NodeUniqueIndexSeek(`idName`, _, Seq(IndexedProperty(_, CanGetValue)), SingleQueryExpression(`lit42`), _, _)) => ()
      }
    }
  }

  test("unique index seek without values when there is an index on the property") {
    new given {
      addTypeToSemanticTable(lit42, CTInteger.invariant)
      qg = queryGraph(lessThanPredicate, hasLabel("Awesome"))

      uniqueIndexOn("Awesome", "prop")
    }.withLogicalPlanningContext { (cfg, ctx) =>
      // when
      val resultPlans = indexSeekLeafPlanner(cfg.qg, InterestingOrder.empty, ctx)

      // then
      resultPlans should beLike {
        case Seq(NodeUniqueIndexSeek(`idName`, _, Seq(IndexedProperty(_, DoNotGetValue)), _, _, _)) => ()
      }
    }
  }

  test("unique index seek with values (from index) when there is an index on the property") {
    new given {
      addTypeToSemanticTable(lit42, CTInteger.invariant)
      qg = queryGraph(lessThanPredicate, hasLabel("Awesome"))

      uniqueIndexOn("Awesome", "prop").providesValues()
    }.withLogicalPlanningContext { (cfg, ctx) =>
      // when
      val resultPlans = indexSeekLeafPlanner(cfg.qg, InterestingOrder.empty, ctx)

      // then
      resultPlans should beLike {
        case Seq(NodeUniqueIndexSeek(`idName`, _, Seq(IndexedProperty(_, CanGetValue)), _, _, _)) => ()
      }
    }
  }

  test("plans index seeks such that it solves hints") {
    val hint: UsingIndexHint = UsingIndexHint(varFor("n"), labelName("Awesome"), Seq(PropertyKeyName("prop")(pos))) _

    new given {
      addTypeToSemanticTable(lit42, CTInteger.invariant)
      qg = queryGraph(inPredicate, hasLabel("Awesome")).addHints(Some(hint))

      indexOn("Awesome", "prop")
    }.withLogicalPlanningContext { (cfg, ctx) =>
      // when
      val resultPlans = indexSeekLeafPlanner(cfg.qg, InterestingOrder.empty, ctx)

      // then
      resultPlans should beLike {
        case Seq(NodeIndexSeek(`idName`, _, _, SingleQueryExpression(`lit42`), _, _)) => ()
      }

      resultPlans.map(p => ctx.planningAttributes.solveds.get(p.id).asSinglePlannerQuery.queryGraph) should beLike {
        case Seq(plannedQG: QueryGraph) if plannedQG.hints == Set(hint) => ()
      }
    }
  }

  test("plans unique index seeks such that it solves hints") {
    val hint: UsingIndexHint = UsingIndexHint(varFor("n"), labelName("Awesome"), Seq(PropertyKeyName("prop")(pos))) _

    new given {
      addTypeToSemanticTable(lit42, CTInteger.invariant)
      qg = queryGraph(inPredicate, hasLabel("Awesome")).addHints(Some(hint))

      uniqueIndexOn("Awesome", "prop")
    }.withLogicalPlanningContext { (cfg, ctx) =>
      // when
      val resultPlans = indexSeekLeafPlanner(cfg.qg, InterestingOrder.empty, ctx)

      // then
      resultPlans should beLike {
        case Seq(NodeUniqueIndexSeek(`idName`, _, _, SingleQueryExpression(`lit42`), _, _)) => ()
      }

      resultPlans.map(p => ctx.planningAttributes.solveds.get(p.id).asSinglePlannerQuery.queryGraph) should beLike {
        case Seq(plannedQG: QueryGraph) if plannedQG.hints == Set(hint) => ()
      }
    }
  }

  test("plans merge unique index seeks when there are two unique indexes") {
    new given {
      addTypeToSemanticTable(lit42, CTInteger.invariant)
      qg = queryGraph(inPredicate, hasLabel("Awesome"), hasLabel("Awesomer"))

      uniqueIndexOn("Awesome", "prop")
      uniqueIndexOn("Awesomer", "prop")
    }.withLogicalPlanningContext { (cfg, ctx) =>
      // when
      val resultPlans = mergeUniqueIndexSeekLeafPlanner(cfg.qg, InterestingOrder.empty, ctx)

      // then
      resultPlans should beLike {
        case Seq(AssertSameNode(`idName`,
        NodeUniqueIndexSeek(`idName`, LabelToken("Awesome", _), _, SingleQueryExpression(`lit42`), _, _),
        NodeUniqueIndexSeek(`idName`, LabelToken("Awesomer", _), _, SingleQueryExpression(`lit42`), _, _))) => ()
      }
    }
  }

  test("plans merge unique index seeks when there are only one unique index") {
    new given {
      addTypeToSemanticTable(lit42, CTInteger.invariant)
      qg = queryGraph(inPredicate, hasLabel("Awesome"), hasLabel("Awesomer"))

      uniqueIndexOn("Awesome", "prop")
    }.withLogicalPlanningContext { (cfg, ctx) =>
      // when
      val resultPlans = mergeUniqueIndexSeekLeafPlanner(cfg.qg, InterestingOrder.empty, ctx)

      // then
      resultPlans should beLike {
        case Seq(NodeUniqueIndexSeek(`idName`, _, _, SingleQueryExpression(`lit42`), _, _)) => ()
      }
    }
  }

  test("plans merge unique index seeks with AssertSameNode when there are three unique indexes") {
    new given {
      addTypeToSemanticTable(lit42, CTInteger.invariant)
      qg = queryGraph(inPredicate, hasLabel("Awesome"), hasLabel("Awesomer"), hasLabel("Awesomest"))

      uniqueIndexOn("Awesome", "prop")
      uniqueIndexOn("Awesomer", "prop")
      uniqueIndexOn("Awesomest", "prop")
    }.withLogicalPlanningContext { (cfg, ctx) =>
      // when
      val resultPlans = mergeUniqueIndexSeekLeafPlanner(cfg.qg, InterestingOrder.empty, ctx)

      // then
      resultPlans should beLike {
        case Seq(
        AssertSameNode(`idName`,
        AssertSameNode(`idName`,
        NodeUniqueIndexSeek(`idName`, LabelToken("Awesome", _), _, SingleQueryExpression(`lit42`), _, _),
        NodeUniqueIndexSeek(`idName`, LabelToken("Awesomer", _), _, SingleQueryExpression(`lit42`), _, _)),
        NodeUniqueIndexSeek(`idName`, LabelToken("Awesomest", _), _, SingleQueryExpression(`lit42`), _, _))) => ()
      }
    }
  }

  test("plans merge unique index seeks with AssertSameNode when there are four unique indexes") {
    new given {
      addTypeToSemanticTable(lit42, CTInteger.invariant)
      qg = queryGraph(inPredicate, hasLabel("Awesome"), hasLabel("Awesomer"),
        hasLabel("Awesomest"), hasLabel("Awesomestest"))

      uniqueIndexOn("Awesome", "prop")
      uniqueIndexOn("Awesomer", "prop")
      uniqueIndexOn("Awesomest", "prop")
      uniqueIndexOn("Awesomestest", "prop")
    }.withLogicalPlanningContext { (cfg, ctx) =>
      // when
      val resultPlans: Seq[LogicalPlan] = mergeUniqueIndexSeekLeafPlanner(cfg.qg, InterestingOrder.empty, ctx)

      // then
      resultPlans should beLike {
        case Seq(
        AssertSameNode(`idName`,
        AssertSameNode(`idName`,
        AssertSameNode(`idName`,
        NodeUniqueIndexSeek(`idName`, LabelToken(l1, _), _, SingleQueryExpression(`lit42`), _, _),
        NodeUniqueIndexSeek(`idName`, LabelToken(l2, _), _, SingleQueryExpression(`lit42`), _, _)),
        NodeUniqueIndexSeek(`idName`, LabelToken(l3, _), _, SingleQueryExpression(`lit42`), _, _)),
        NodeUniqueIndexSeek(`idName`, LabelToken(l4, _), _, SingleQueryExpression(`lit42`), _, _)))
          if Set(l1, l2, l3, l4) == Set("Awesome", "Awesomer", "Awesomest", "Awesomestest") => ()
      }
    }
  }

  test("test with three predicates, a single prop constraint and a two-prop constraint") {
    // MERGE (a:X {prop1: 42, prop2: 444, prop3: 56})
    // Unique constraint on :X(prop1, prop2)
    // Unique constraint on :X(prop3)

    val val1 = literalInt(44)
    val val2 = literalInt(55)
    val val3 = literalInt(66)
    val pred1 = equals(prop("n", "prop1"), val1)
    val pred2 = equals(prop("n", "prop2"), val2)
    val pred3 = equals(prop("n", "prop3"), val3)
    new given {
      addTypeToSemanticTable(val1, CTInteger.invariant)
      addTypeToSemanticTable(val2, CTInteger.invariant)
      addTypeToSemanticTable(val3, CTInteger.invariant)
      qg = queryGraph(pred1, pred2, pred3, hasLabel("Awesome"))

      uniqueIndexOn("Awesome", "prop1", "prop2")
      uniqueIndexOn("Awesome", "prop3")
    }.withLogicalPlanningContext { (cfg, ctx) =>
      // when
      val resultPlans = mergeUniqueIndexSeekLeafPlanner(cfg.qg, InterestingOrder.empty, ctx)

      // then
      resultPlans should beLike {
        case Seq(
        AssertSameNode(`idName`,
        NodeUniqueIndexSeek(`idName`, LabelToken("Awesome", _), Seq(IndexedProperty(PropertyKeyToken("prop1", _), CanGetValue), IndexedProperty(PropertyKeyToken("prop2", _), CanGetValue)),
        CompositeQueryExpression(Seq(
        SingleQueryExpression(`val1`),
        SingleQueryExpression(`val2`))), _, _),
        NodeUniqueIndexSeek(`idName`, LabelToken("Awesome", _), _,
        SingleQueryExpression(`val3`), _, _))) => ()
      }
    }
  }

  test("test with three predicates, two composite two-prop constraints") {
    // MERGE (a:X {prop1: 42, prop2: 444, prop3: 56})
    // Unique constraint on :X(prop1, prop2)
    // Unique constraint on :X(prop2, prop3)

    val val1 = literalInt(44)
    val val2 = literalInt(55)
    val val3 = literalInt(66)
    val pred1 = equals(prop("n", "prop1"), val1)
    val pred2 = equals(prop("n", "prop2"), val2)
    val pred3 = equals(prop("n", "prop3"), val3)
    new given {
      addTypeToSemanticTable(val1, CTInteger.invariant)
      addTypeToSemanticTable(val2, CTInteger.invariant)
      addTypeToSemanticTable(val3, CTInteger.invariant)
      qg = queryGraph(pred1, pred2, pred3, hasLabel("Awesome"))

      uniqueIndexOn("Awesome", "prop1", "prop2")
      uniqueIndexOn("Awesome", "prop2", "prop3")
    }.withLogicalPlanningContext { (cfg, ctx) =>
      // when
      val resultPlans = mergeUniqueIndexSeekLeafPlanner(cfg.qg, InterestingOrder.empty, ctx)

      // then
      resultPlans should beLike {
        case Seq(
        AssertSameNode(`idName`,
        NodeUniqueIndexSeek(`idName`, LabelToken("Awesome", _), Seq(IndexedProperty(PropertyKeyToken("prop1", _), CanGetValue), IndexedProperty(PropertyKeyToken("prop2", _), CanGetValue)),
        CompositeQueryExpression(Seq(
        SingleQueryExpression(`val1`),
        SingleQueryExpression(`val2`))), _, _),
        NodeUniqueIndexSeek(`idName`, LabelToken("Awesome", _), Seq(IndexedProperty(PropertyKeyToken("prop2", _), CanGetValue), IndexedProperty(PropertyKeyToken("prop3", _), CanGetValue)),
        CompositeQueryExpression(Seq(
        SingleQueryExpression(`val2`),
        SingleQueryExpression(`val3`))), _, _))) => ()
      }
    }
  }

  test("test with three predicates, single composite three-prop constraints") {
    // MERGE (a:X {prop1: 42, prop2: 444, prop3: 56})
    // Unique constraint on :X(prop1, prop2, prop3)

    val val1 = literalInt(44)
    val val2 = literalInt(55)
    val val3 = literalInt(66)
    val pred1 = equals(prop("n", "prop1"), val1)
    val pred2 = equals(prop("n", "prop2"), val2)
    val pred3 = equals(prop("n", "prop3"), val3)
    new given {
      addTypeToSemanticTable(val1, CTInteger.invariant)
      addTypeToSemanticTable(val2, CTInteger.invariant)
      addTypeToSemanticTable(val3, CTInteger.invariant)
      qg = queryGraph(pred1, pred2, pred3, hasLabel("Awesome"))

      uniqueIndexOn("Awesome", "prop1", "prop2", "prop3")
    }.withLogicalPlanningContext { (cfg, ctx) =>
      // when
      val resultPlans = mergeUniqueIndexSeekLeafPlanner(cfg.qg, InterestingOrder.empty, ctx)

      // then
      resultPlans should beLike {
        case Seq(
        NodeUniqueIndexSeek(`idName`, LabelToken("Awesome", _),
        Seq(IndexedProperty(PropertyKeyToken("prop1", _), CanGetValue), IndexedProperty(PropertyKeyToken("prop2", _), CanGetValue), IndexedProperty(PropertyKeyToken("prop3", _), CanGetValue)),
        CompositeQueryExpression(Seq(
        SingleQueryExpression(`val1`),
        SingleQueryExpression(`val2`),
        SingleQueryExpression(`val3`))), _, _)
        ) => ()
      }
    }
  }

  private def queryGraph(predicates: Expression*) =
    QueryGraph(
      selections = Selections(predicates.map(Predicate(Set(idName), _)).toSet),
      patternNodes = Set(idName)
    )
}
