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
package org.neo4j.cypher.internal.runtime.interpreted.pipes

import org.mockito.Mockito.verify
import org.mockito.Mockito.when
import org.neo4j.cypher.internal.runtime.ClosingIterator
import org.neo4j.cypher.internal.runtime.CypherRow
import org.neo4j.cypher.internal.runtime.MapCypherRow
import org.neo4j.cypher.internal.runtime.QueryContext
import org.neo4j.cypher.internal.util.test_helpers.CypherFunSuite
import org.neo4j.exceptions.CypherTypeException
import org.neo4j.values.AnyValue
import org.neo4j.values.storable.Values
import org.neo4j.values.virtual.VirtualValues

import scala.collection.mutable

class LockNodesPipeTest extends CypherFunSuite {


  test("should handle NoValue input") {
    // given
    val state = mock[QueryState]
    val queryContext = mock[QueryContext]
    when(state.query).thenReturn(queryContext)

    val source = iteratorWithValues(VirtualValues.node(1), Values.NO_VALUE, VirtualValues.node(2), Values.NO_VALUE)

    // when
    val x = LockNodesPipe(mock[Pipe], Set("x"))()
    val result = x.testCreateResults(ClosingIterator(source), state)

    // then
    result.next()
    verify(queryContext).lockNodes(1)

    result.next()
    verify(queryContext).lockNodes(1)

    result.next()
    verify(queryContext).lockNodes(2)

    result.next()
    verify(queryContext).lockNodes(2)
  }

  test("should crash with CypherTypeException on illegal value types") {
    // given
    val state = mock[QueryState]
    val queryContext = mock[QueryContext]
    when(state.query).thenReturn(queryContext)

    for (illegal <- List(VirtualValues.relationship(1), Values.of(1), Values.of("hi"))) {
      val source = iteratorWithValues(illegal)

      // when
      val x = LockNodesPipe(mock[Pipe], Set("x"))()
      val result = x.testCreateResults(ClosingIterator(source), state)

      // then
      intercept[CypherTypeException] { result.next() }
    }
  }

  private def iteratorWithValues(values: AnyValue*): Iterator[CypherRow] = {
    values.map(rowWithValue).iterator
  }

  private def rowWithValue(value: AnyValue) = {
    new MapCypherRow(mutable.Map("x" -> value), mutable.Map.empty)
  }
}
