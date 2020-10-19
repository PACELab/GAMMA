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
package org.neo4j.cypher.internal.compiler

import org.neo4j.cypher.internal.ast.CreateIndex
import org.neo4j.cypher.internal.ast.CreateIndexNewSyntax
import org.neo4j.cypher.internal.ast.CreateNodeKeyConstraint
import org.neo4j.cypher.internal.ast.CreateNodePropertyExistenceConstraint
import org.neo4j.cypher.internal.ast.CreateRelationshipPropertyExistenceConstraint
import org.neo4j.cypher.internal.ast.CreateUniquePropertyConstraint
import org.neo4j.cypher.internal.ast.DropConstraintOnName
import org.neo4j.cypher.internal.ast.DropIndex
import org.neo4j.cypher.internal.ast.DropIndexOnName
import org.neo4j.cypher.internal.ast.DropNodeKeyConstraint
import org.neo4j.cypher.internal.ast.DropNodePropertyExistenceConstraint
import org.neo4j.cypher.internal.ast.DropRelationshipPropertyExistenceConstraint
import org.neo4j.cypher.internal.ast.DropUniquePropertyConstraint
import org.neo4j.cypher.internal.ast.IfExistsDoNothing
import org.neo4j.cypher.internal.ast.IfExistsReplace
import org.neo4j.cypher.internal.compiler.phases.LogicalPlanState
import org.neo4j.cypher.internal.compiler.phases.PlannerContext
import org.neo4j.cypher.internal.frontend.phases.BaseState
import org.neo4j.cypher.internal.frontend.phases.CompilationPhaseTracer.CompilationPhase
import org.neo4j.cypher.internal.frontend.phases.CompilationPhaseTracer.CompilationPhase.PIPE_BUILDING
import org.neo4j.cypher.internal.frontend.phases.Condition
import org.neo4j.cypher.internal.frontend.phases.Phase
import org.neo4j.cypher.internal.logical.plans
import org.neo4j.cypher.internal.logical.plans.LogicalPlan
import org.neo4j.cypher.internal.planner.spi.AdministrationPlannerName
import org.neo4j.cypher.internal.util.attribution.SequentialIdGen

/**
 * This planner takes on queries that requires no planning such as schema commands
 */
case object SchemaCommandPlanBuilder extends Phase[PlannerContext, BaseState, LogicalPlanState] {

  override def phase: CompilationPhase = PIPE_BUILDING

  override def description = "take on queries that require no planning such as schema commands"

  override def postConditions: Set[Condition] = Set.empty

  override def process(from: BaseState, context: PlannerContext): LogicalPlanState = {
    implicit val idGen = new SequentialIdGen()
    val maybeLogicalPlan: Option[LogicalPlan] = from.statement() match {
      // CREATE CONSTRAINT ON (node:Label) ASSERT (node.prop1,node.prop2) IS NODE KEY
      case CreateNodeKeyConstraint(node, label, props, name, ifExistsDo, _) =>
        val source = ifExistsDo match {
          case IfExistsReplace =>
            // Name is not optional with OR REPLACE
            // This has been checked in semantic checking, so it is safe to call name.get now
            Some(plans.DropConstraintOnName(name.get, ifExists = true))
          case IfExistsDoNothing => Some(plans.DoNothingIfExistsForConstraint(node.name, scala.util.Left(label), props, plans.NodeKey, name))
          case _ => None
        }
        Some(plans.CreateNodeKeyConstraint(source, node.name, label, props, name))

      // DROP CONSTRAINT ON (node:Label) ASSERT (node.prop1,node.prop2) IS NODE KEY
      case DropNodeKeyConstraint(_, label, props, _) =>
        Some(plans.DropNodeKeyConstraint(label, props))

      // CREATE CONSTRAINT ON (node:Label) ASSERT node.prop IS UNIQUE
      // CREATE CONSTRAINT ON (node:Label) ASSERT (node.prop1,node.prop2) IS UNIQUE
      case CreateUniquePropertyConstraint(node, label, props, name, ifExistsDo, _) =>
        val source = ifExistsDo match {
          case IfExistsReplace =>
            // Name is not optional with OR REPLACE
            // This has been checked in semantic checking, so it is safe to call name.get now
            Some(plans.DropConstraintOnName(name.get, ifExists = true))
          case IfExistsDoNothing => Some(plans.DoNothingIfExistsForConstraint(node.name, scala.util.Left(label), props, plans.Uniqueness, name))
          case _ => None
        }
        Some(plans.CreateUniquePropertyConstraint(source, node.name, label, props, name))

      // DROP CONSTRAINT ON (node:Label) ASSERT node.prop IS UNIQUE
      // DROP CONSTRAINT ON (node:Label) ASSERT (node.prop1,node.prop2) IS UNIQUE
      case DropUniquePropertyConstraint(_, label, props, _) =>
        Some(plans.DropUniquePropertyConstraint(label, props))

      // CREATE CONSTRAINT ON (node:Label) ASSERT node.prop EXISTS
      case CreateNodePropertyExistenceConstraint(_, label, prop, name, ifExistsDo, _) =>
        val source = ifExistsDo match {
          case IfExistsReplace =>
            // Name is not optional with OR REPLACE
            // This has been checked in semantic checking, so it is safe to call name.get now
            Some(plans.DropConstraintOnName(name.get, ifExists = true))
          case IfExistsDoNothing => Some(plans.DoNothingIfExistsForConstraint(prop.map.asCanonicalStringVal, scala.util.Left(label), Seq(prop), plans.NodePropertyExistence, name))
          case _ => None
        }
        Some(plans.CreateNodePropertyExistenceConstraint(source, label, prop, name))

      // DROP CONSTRAINT ON (node:Label) ASSERT node.prop EXISTS
      case DropNodePropertyExistenceConstraint(_, label, prop, _) =>
        Some(plans.DropNodePropertyExistenceConstraint(label, prop))

      // CREATE CONSTRAINT ON ()-[r:R]-() ASSERT r.prop EXISTS
      case CreateRelationshipPropertyExistenceConstraint(_, relType, prop, name, ifExistsDo, _) =>
        val source = ifExistsDo match {
          case IfExistsReplace =>
            // Name is not optional with OR REPLACE
            // This has been checked in semantic checking, so it is safe to call name.get now
            Some(plans.DropConstraintOnName(name.get, ifExists = true))
          case IfExistsDoNothing => Some(plans.DoNothingIfExistsForConstraint(prop.map.asCanonicalStringVal, scala.util.Right(relType), Seq(prop), plans.RelationshipPropertyExistence, name))
          case _ => None
        }
        Some(plans.CreateRelationshipPropertyExistenceConstraint(source, relType, prop, name))

      // DROP CONSTRAINT ON ()-[r:R]-() ASSERT r.prop EXISTS
      case DropRelationshipPropertyExistenceConstraint(_, relType, prop, _) =>
        Some(plans.DropRelationshipPropertyExistenceConstraint(relType, prop))

      // DROP CONSTRAINT name
      case DropConstraintOnName(name, ifExists, _) =>
        Some(plans.DropConstraintOnName(name, ifExists))

      // CREATE INDEX ON :LABEL(prop)
      case CreateIndex(label, props, _) =>
        Some(plans.CreateIndex(None, label, props, None))

      // CREATE INDEX FOR (n:LABEL) ON (n.prop)
      // CREATE INDEX name FOR (n:LABEL) ON (n.prop)
      case CreateIndexNewSyntax(_, label, props, name, ifExistsDo, _) =>
        val propKeys = props.map(_.propertyKey)
        val source = ifExistsDo match {
          case IfExistsReplace =>
            // Name is not optional with OR REPLACE
            // This has been checked in semantic checking, so it is safe to call name.get now
            Some(plans.DropIndexOnName(name.get, ifExists = true))
          case IfExistsDoNothing => Some(plans.DoNothingIfExistsForIndex(label, propKeys, name))
          case _ => None
        }
        Some(plans.CreateIndex(source, label, propKeys, name))

      // DROP INDEX ON :LABEL(prop)
      case DropIndex(label, props, _) =>
        Some(plans.DropIndex(label, props))

      // DROP INDEX name
      case DropIndexOnName(name, ifExists, _) =>
        Some(plans.DropIndexOnName(name, ifExists))

      case _ => None
    }

    val planState = LogicalPlanState(from)

    if (maybeLogicalPlan.isDefined)
      planState.copy(maybeLogicalPlan = maybeLogicalPlan, plannerName = AdministrationPlannerName)
    else planState
  }
}
