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
package org.neo4j.cypher.internal.procs

import org.neo4j.cypher.internal.ExecutionPlan
import org.neo4j.cypher.internal.runtime.ExecutionMode
import org.neo4j.cypher.internal.runtime.InputDataStream
import org.neo4j.cypher.internal.runtime.QueryContext
import org.neo4j.cypher.internal.runtime.interpreted.UpdateCountingQueryContext
import org.neo4j.cypher.result.RuntimeResult
import org.neo4j.kernel.impl.query.QuerySubscriber
import org.neo4j.values.virtual.MapValue

/**
 * Execution plan for performing schema writes, i.e. creating or dropping indexes and constraints.
 *
 * @param name        A name of the schema write
 * @param schemaWrite The actual schema write to perform
 */
case class SchemaWriteExecutionPlan(name: String, schemaWrite: QueryContext => SchemaWriteExecutionResult, source: Option[ExecutionPlan] = None)
  extends SchemaCommandChainedExecutionPlan(source) {

  override def runSpecific(ctx: UpdateCountingQueryContext,
                   executionMode: ExecutionMode,
                   params: MapValue,
                   prePopulateResults: Boolean,
                   ignore: InputDataStream,
                   subscriber: QuerySubscriber): RuntimeResult = {

    ctx.assertSchemaWritesAllowed()

    if (schemaWrite(ctx) == SuccessResult) {
      ctx.transactionalContext.close()
      val runtimeResult = SchemaWriteRuntimeResult(ctx, subscriber)
      runtimeResult
    } else {
      IgnoredRuntimeResult
    }
  }
}

sealed trait SchemaWriteExecutionResult

case object SuccessResult extends SchemaWriteExecutionResult
case object IgnoredResult extends SchemaWriteExecutionResult
