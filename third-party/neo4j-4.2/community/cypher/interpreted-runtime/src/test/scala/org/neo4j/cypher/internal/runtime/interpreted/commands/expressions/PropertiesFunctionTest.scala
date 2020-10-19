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
package org.neo4j.cypher.internal.runtime.interpreted.commands.expressions

import java.util

import org.mockito.Mockito.when
import org.neo4j.cypher.internal.runtime.CypherRow
import org.neo4j.cypher.internal.runtime.NodeOperations
import org.neo4j.cypher.internal.runtime.QueryContext
import org.neo4j.cypher.internal.runtime.RelationshipOperations
import org.neo4j.cypher.internal.runtime.interpreted.QueryStateHelper
import org.neo4j.cypher.internal.runtime.interpreted.commands.LiteralHelper.literal
import org.neo4j.cypher.internal.util.test_helpers.CypherFunSuite
import org.neo4j.exceptions.CypherTypeException
import org.neo4j.graphdb.Node
import org.neo4j.graphdb.Relationship
import org.neo4j.values.storable.Values.NO_VALUE
import org.neo4j.values.storable.Values.stringValue
import org.neo4j.values.virtual.VirtualValues.map

class PropertiesFunctionTest extends CypherFunSuite {


  val query = mock[QueryContext]
  val nodeOps = mock[NodeOperations]
  val relOps = mock[RelationshipOperations]

  when(query.nodeOps).thenReturn(nodeOps)
  when(query.relationshipOps).thenReturn(relOps)

  test("should return null if argument is null") {
    properties(null.asInstanceOf[AnyRef]) should be(NO_VALUE)
  }

  test("should map Java maps to maps") {
    val m = new util.HashMap[String, String]()
    m.put("a", "x")
    m.put("b", "y")

    properties(m) should equal(map(Array("a", "b"), Array(stringValue("x"), stringValue("y"))))
  }

  test("should map Scala maps to maps") {
    val m = Map("a" -> "x", "b" -> "y")
    properties(m) should equal(map(Array("a", "b"), Array(stringValue("x"), stringValue("y"))))

  }

  test("should map nodes to maps") {
    val node = mock[Node]
    when(node.getId).thenReturn(0)
    val value = map(Array("a", "b"), Array(stringValue("x"), stringValue("y")))
    when(query.nodeAsMap(0, null, null)).thenReturn(value)

    properties(node) should equal(value)
  }

  test("should map relationships to maps") {
    val rel = mock[Relationship]
    when(rel.getId).thenReturn(0)
    val value = map(Array("a", "b"), Array(stringValue("x"), stringValue("y")))
    when(query.relationshipAsMap(0, null, null)).thenReturn(value)

    properties(rel) should equal(value)
  }

  test("should fail trying to map an int") {
    a[CypherTypeException] should be thrownBy {
      properties(12)
    }
  }

  test("should fail trying to map a string") {
    a[CypherTypeException] should be thrownBy {
      properties("Hullo")
    }
  }

  test("should fail trying to map a list") {
    a[CypherTypeException] should be thrownBy {
      properties(List.empty)
    }
  }

  private def properties(orig: Any) = {
    PropertiesFunction(literal(orig))(CypherRow.empty, QueryStateHelper.empty.withQueryContext(query))
  }
}
