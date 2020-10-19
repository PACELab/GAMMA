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
package org.neo4j.cypher.internal.compiler.planner

import org.neo4j.cypher.internal.compiler.MissingLabelNotification
import org.neo4j.cypher.internal.compiler.MissingPropertyNameNotification
import org.neo4j.cypher.internal.compiler.MissingRelTypeNotification
import org.neo4j.cypher.internal.compiler.phases.LogicalPlanState
import org.neo4j.cypher.internal.expressions.LabelName
import org.neo4j.cypher.internal.expressions.Property
import org.neo4j.cypher.internal.expressions.PropertyKeyName
import org.neo4j.cypher.internal.expressions.RelTypeName
import org.neo4j.cypher.internal.frontend.phases.BaseContext
import org.neo4j.cypher.internal.frontend.phases.CompilationPhaseTracer.CompilationPhase.LOGICAL_PLANNING
import org.neo4j.cypher.internal.frontend.phases.VisitorPhase
import org.neo4j.cypher.internal.util.Foldable.TraverseChildren
import org.neo4j.cypher.internal.util.InternalNotification
import org.neo4j.values.storable.DurationFields
import org.neo4j.values.storable.PointFields
import org.neo4j.values.storable.TemporalValue.TemporalFields

import scala.collection.JavaConverters.asScalaSetConverter

object CheckForUnresolvedTokens extends VisitorPhase[BaseContext, LogicalPlanState] {

  private val specialPropertyKey: Set[String] =
    (TemporalFields.allFields().asScala ++
      DurationFields.values().map(_.propertyKey) ++
      PointFields.values().map(_.propertyKey)).map(_.toLowerCase).toSet

  override def visit(value: LogicalPlanState, context: BaseContext): Unit = {
    val table = value.semanticTable()
    def isEmptyLabel(label: String) = !table.resolvedLabelNames.contains(label)
    def isEmptyRelType(relType: String) = !table.resolvedRelTypeNames.contains(relType)
    def isEmptyPropertyName(name: String) = !table.resolvedPropertyKeyNames.contains(name)

    val notifications = value.statement().treeFold(Seq.empty[InternalNotification]) {
      case label@LabelName(name) if isEmptyLabel(name) => acc =>
        TraverseChildren(acc :+ MissingLabelNotification(label.position, name))

      case rel@RelTypeName(name) if isEmptyRelType(name) => acc =>
        TraverseChildren(acc :+ MissingRelTypeNotification(rel.position, name))

      case Property(_, prop@PropertyKeyName(name)) if !specialPropertyKey(name.toLowerCase) && isEmptyPropertyName(name) => acc =>
        TraverseChildren(acc :+ MissingPropertyNameNotification(prop.position, name))
    }

    notifications foreach context.notificationLogger.log
  }

  override def phase = LOGICAL_PLANNING

  override def description = "find labels, relationships types and property keys that do not exist in the db and issue warnings"
}
