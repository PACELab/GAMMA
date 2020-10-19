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

import org.neo4j.cypher.internal.logical.plans.IndexOrder
import org.neo4j.cypher.internal.runtime.ClosingIterator
import org.neo4j.cypher.internal.runtime.CypherRow
import org.neo4j.cypher.internal.runtime.interpreted.pipes.LazyLabel.UNKNOWN
import org.neo4j.cypher.internal.util.attribution.Id

case class NodeByLabelScanPipe(ident: String, label: LazyLabel, indexOrder: IndexOrder)
                              (val id: Id = Id.INVALID_ID) extends Pipe {

  protected def internalCreateResults(state: QueryState): ClosingIterator[CypherRow] = {

    val id = label.getId(state.query)
    if (id != UNKNOWN) {
      val nodes = state.query.getNodesByLabel(id, indexOrder)
      val baseContext = state.newRowWithArgument(rowFactory)
      nodes.map(n => rowFactory.copyWith(baseContext, ident, n))
    } else {
      ClosingIterator.empty
    }
  }
}
