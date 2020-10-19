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

import org.neo4j.cypher.internal.runtime.ClosingIterator
import org.neo4j.internal.kernel.api.NodeValueIndexCursor

import scala.collection.Iterator

abstract class IndexIteratorBase[T](state: QueryState, val cursor: NodeValueIndexCursor) extends ClosingIterator[T] {
  private var _next: T = fetchNext()

  protected def fetchNext(): T

  protected def closeMore(): Unit = cursor.close()

  override final def innerHasNext: Boolean = _next != null

  override final def next(): T = {
    if (!hasNext) {
      Iterator.empty.next()
    }

    val current = _next
    _next = fetchNext()
    current
  }
}
