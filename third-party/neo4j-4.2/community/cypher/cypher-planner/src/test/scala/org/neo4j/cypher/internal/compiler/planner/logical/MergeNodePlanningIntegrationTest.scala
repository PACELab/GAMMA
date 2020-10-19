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
package org.neo4j.cypher.internal.compiler.planner.logical

import org.neo4j.cypher.internal.compiler.planner.LogicalPlanningTestSupport2
import org.neo4j.cypher.internal.expressions.PropertyKeyName
import org.neo4j.cypher.internal.ir.CreateNode
import org.neo4j.cypher.internal.logical.plans.AllNodesScan
import org.neo4j.cypher.internal.logical.plans.AntiConditionalApply
import org.neo4j.cypher.internal.logical.plans.Apply
import org.neo4j.cypher.internal.logical.plans.Argument
import org.neo4j.cypher.internal.logical.plans.AssertSameNode
import org.neo4j.cypher.internal.logical.plans.ConditionalApply
import org.neo4j.cypher.internal.logical.plans.Create
import org.neo4j.cypher.internal.logical.plans.Eager
import org.neo4j.cypher.internal.logical.plans.EmptyResult
import org.neo4j.cypher.internal.logical.plans.IndexOrderNone
import org.neo4j.cypher.internal.logical.plans.MergeCreateNode
import org.neo4j.cypher.internal.logical.plans.NodeByLabelScan
import org.neo4j.cypher.internal.logical.plans.NodeUniqueIndexSeek
import org.neo4j.cypher.internal.logical.plans.Optional
import org.neo4j.cypher.internal.logical.plans.Selection
import org.neo4j.cypher.internal.logical.plans.SetLabels
import org.neo4j.cypher.internal.logical.plans.SetNodeProperty
import org.neo4j.cypher.internal.util.test_helpers.CypherFunSuite

class MergeNodePlanningIntegrationTest extends CypherFunSuite with LogicalPlanningTestSupport2 {
  test("should plan single merge node") {
    val allNodesScan = AllNodesScan("a", Set.empty)
    val optional = Optional(allNodesScan)
    val onCreate = MergeCreateNode(Argument(), "a", Seq.empty, None)

    val mergeNode = AntiConditionalApply(optional, onCreate, Seq("a"))
    val emptyResult = EmptyResult(mergeNode)

    planFor("MERGE (a)")._2 should equal(emptyResult)
  }

  test("should plan single merge node from a label scan") {

    val labelScan = NodeByLabelScan("a", labelName("X"), Set.empty, IndexOrderNone)
    val optional = Optional(labelScan)
    val onCreate = MergeCreateNode(Argument(), "a", Seq(labelName("X")), None)

    val mergeNode = AntiConditionalApply(optional, onCreate, Seq("a"))
    val emptyResult = EmptyResult(mergeNode)

    (new given {
      labelCardinality = Map(
        "X" -> 30.0
      )
    } getLogicalPlanFor "MERGE (a:X)")._2 should equal(emptyResult)
  }

  test("should plan single merge node with properties") {

    val allNodesScan = AllNodesScan("a", Set.empty)
    val selection = Selection(Seq(equals(prop("a", "prop"), literalInt(42))), allNodesScan)
    val optional = Optional(selection)

    val onCreate = MergeCreateNode(Argument(), "a", Seq.empty, Some(mapOfInt(("prop", 42))))

    val mergeNode = AntiConditionalApply(optional, onCreate, Seq("a"))
    val emptyResult = EmptyResult(mergeNode)

    planFor("MERGE (a {prop: 42})")._2 should equal(emptyResult)
  }

  test("should plan create followed by merge") {
    val createNode = Create(Argument(), nodes = List(CreateNode("a", Seq.empty, None)), Nil)
    val allNodesScan = AllNodesScan("b", Set.empty)
    val optional = Optional(allNodesScan)
    val onCreate = MergeCreateNode(Argument(), "b", Seq.empty, None)
    val mergeNode = AntiConditionalApply(optional, onCreate, Seq("b"))
    val apply = Apply(createNode, mergeNode)
    val emptyResult = EmptyResult(apply)

    planFor("CREATE (a) MERGE (b)")._2 should equal(emptyResult)
  }

  test("should plan merge followed by create") {
    val allNodesScan = AllNodesScan("a", Set.empty)
    val optional = Optional(allNodesScan)
    val onCreate = MergeCreateNode(Argument(), "a", Seq.empty, None)
    val mergeNode = AntiConditionalApply(optional, onCreate, Seq("a"))
    val eager = Eager(mergeNode)
    val createNode = Create(eager, nodes = List(CreateNode("b", Seq.empty, None)), Nil)
    val emptyResult = EmptyResult(createNode)

    planFor("MERGE(a) CREATE (b)")._2 should equal(emptyResult)
  }

  test("should use AssertSameNode when multiple unique index matches") {
    val plan = (new given {
      uniqueIndexOn("X", "prop")
      uniqueIndexOn("Y", "prop")
    } getLogicalPlanFor "MERGE (a:X:Y {prop: 42})")._2

    plan shouldBe using[AssertSameNode]
    plan shouldBe using[NodeUniqueIndexSeek]
  }

  test("should not use AssertSameNode when one unique index matches") {
    val plan = (new given {
      uniqueIndexOn("X", "prop")
    } getLogicalPlanFor "MERGE (a:X:Y {prop: 42})")._2

    plan should not be using[AssertSameNode]
    plan shouldBe using[NodeUniqueIndexSeek]
  }

  /*
   *                     |
   *                antiCondApply
   *                  /    \
   *                 /  set property
   *                /       \
   *               /    merge create node
   *         condApply       \
   *            /    \       arg2
   *       optional  set label
   *         /        \
   *    allnodes     arg1
   */
  test("should plan merge node with on create and on match ") {
    val allNodesScan = AllNodesScan("a", Set.empty)
    val optional = Optional(allNodesScan)
    val argument1 = Argument(Set("a"))
    val setLabels = SetLabels(argument1, "a", Seq(labelName("L")))
    val onMatch = ConditionalApply(optional, setLabels, Seq("a"))

    val argument2 = Argument()
    val mergeCreateNode = MergeCreateNode(argument2, "a", Seq.empty, None)
    val createAndOnCreate = SetNodeProperty(mergeCreateNode, "a", PropertyKeyName("prop")(pos), literalInt(1))
    val mergeNode = AntiConditionalApply(onMatch, createAndOnCreate, Seq("a"))
    val emptyResult = EmptyResult(mergeNode)

    planFor("MERGE (a) ON CREATE SET a.prop = 1 ON MATCH SET a:L")._2 should equal(emptyResult)
  }
}
