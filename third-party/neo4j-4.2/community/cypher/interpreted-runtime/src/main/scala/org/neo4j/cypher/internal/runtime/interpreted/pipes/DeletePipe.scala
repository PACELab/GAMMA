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
import org.neo4j.cypher.internal.runtime.IsNoValue
import org.neo4j.cypher.internal.runtime.interpreted.GraphElementPropertyFunctions
import org.neo4j.cypher.internal.runtime.interpreted.commands.expressions.Expression
import org.neo4j.cypher.internal.util.attribution.Id
import org.neo4j.exceptions.CypherTypeException
import org.neo4j.values.virtual.NodeValue
import org.neo4j.values.virtual.PathValue
import org.neo4j.values.virtual.RelationshipValue

case class DeletePipe(src: Pipe, expression: Expression, forced: Boolean)
                     (val id: Id = Id.INVALID_ID)
  extends PipeWithSource(src) with GraphElementPropertyFunctions {

  override protected def internalCreateResults(input: ClosingIterator[CypherRow], state: QueryState): ClosingIterator[CypherRow] = {
    input.map { row =>
      expression(row, state) match {
        case IsNoValue() => // do nothing
        case r: RelationshipValue =>
          deleteRelationship(r, state)
        case n: NodeValue =>
          deleteNode(n, state)
        case p: PathValue =>
          deletePath(p, state)
        case other =>
          throw new CypherTypeException(s"Expected a Node, Relationship or Path, but got a ${other.getClass.getSimpleName}")
      }
      row
    }
  }

  private def deleteNode(n: NodeValue, state: QueryState) = if (!state.query.nodeOps.isDeletedInThisTx(n.id())) {
    if (forced) state.query.detachDeleteNode(n.id())
    else state.query.nodeOps.delete(n.id())
  }

  private def deleteRelationship(r: RelationshipValue, state: QueryState): Unit =
    if (!state.query.relationshipOps.isDeletedInThisTx(r.id())) state.query.relationshipOps.delete(r.id())

  private def deletePath(p: PathValue, state: QueryState): Unit = {
    val entities = p.asList().iterator()
    while (entities.hasNext) {
       entities.next() match {
        case n: NodeValue =>
          deleteNode(n, state)
        case r: RelationshipValue =>
          deleteRelationship(r, state)
        case other =>
          throw new CypherTypeException(s"Expected a Node or Relationship, but got a ${other.getClass.getSimpleName}")
      }
    }
  }
}
