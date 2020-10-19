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
package org.neo4j.cypher.internal.expressions

import org.neo4j.cypher.internal.util.Foldable
import org.neo4j.cypher.internal.util.InputPosition
import org.neo4j.cypher.internal.util.Rewritable
import org.neo4j.cypher.internal.util.Rewritable.IteratorEq

sealed trait PathStep extends Product with Foldable with Rewritable {

  self =>

  def dependencies: Set[LogicalVariable]

  def dup(children: Seq[AnyRef]): this.type =
    if (children.iterator eqElements this.children)
      this
    else {
      val constructor = Rewritable.copyConstructor(this)
      val args = children.toVector
      val ctorArgs = args
      val duped = constructor.invoke(this, ctorArgs: _*)
      duped.asInstanceOf[self.type]
    }
}

final case class NodePathStep(node: LogicalVariable, next: PathStep) extends PathStep {
  val dependencies = next.dependencies + node
}

final case class SingleRelationshipPathStep(rel: LogicalVariable, direction: SemanticDirection, toNode: Option[LogicalVariable], next: PathStep) extends PathStep {
  val dependencies = next.dependencies + rel
}

final case class MultiRelationshipPathStep(rel: LogicalVariable, direction: SemanticDirection, toNode: Option[LogicalVariable], next: PathStep) extends PathStep {
  val dependencies = next.dependencies + rel
}

case object NilPathStep extends PathStep {
  def dependencies = Set.empty[LogicalVariable]
}

case class PathExpression(step: PathStep)(val position: InputPosition) extends Expression
