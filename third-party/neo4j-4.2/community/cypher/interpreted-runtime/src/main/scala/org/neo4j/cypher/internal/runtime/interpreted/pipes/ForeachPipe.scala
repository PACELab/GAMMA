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
import org.neo4j.cypher.internal.runtime.CypherRow
import org.neo4j.cypher.internal.runtime.ListSupport
import org.neo4j.cypher.internal.runtime.interpreted.commands.expressions.Expression
import org.neo4j.cypher.internal.util.attribution.Id

case class ForeachPipe(source: Pipe, inner: Pipe, variable: String, expression: Expression)
                      (val id: Id = Id.INVALID_ID)
  extends PipeWithSource(source) with ListSupport {

  override protected def internalCreateResults(input: ClosingIterator[CypherRow], state: QueryState): ClosingIterator[CypherRow] =
    input.map {
      outerContext =>
        val values = makeTraversable(expression(outerContext, state)).iterator()
        while (values.hasNext) {
          val innerState = state.withInitialContext(rowFactory.copyWith(outerContext, variable, values.next()))
          inner.createResults(innerState).length // exhaust the iterator, in case there's a merge read increasing cardinality inside the foreach
        }
        outerContext
    }
}
