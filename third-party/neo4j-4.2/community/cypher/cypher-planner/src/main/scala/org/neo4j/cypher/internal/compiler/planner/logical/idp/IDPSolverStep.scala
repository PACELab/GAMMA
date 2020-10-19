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
package org.neo4j.cypher.internal.compiler.planner.logical.idp

import scala.collection.GenTraversableOnce

object IDPSolverStep {
  def empty[S, O, P, C] = new IDPSolverStep[S, O, P, C] {
    override def apply(registry: IdRegistry[S], goal: Goal, cache: IDPCache[P, O], context: C): Iterator[P] =
      Iterator.empty
  }
}

trait SolverStep[S, O, P, C] {
  def apply(registry: IdRegistry[S], goal: Goal, cache: IDPCache[P, O], context: C): Iterator[P]
}

trait IDPSolverStep[S, O, P, C] extends SolverStep[S, O, P, C] {
  self =>

  def map(f: P => P): IDPSolverStep[S, O, P, C] = new IDPSolverStep[S, O, P, C] {
    override def apply(registry: IdRegistry[S], goal: Goal, cache: IDPCache[P, O], context: C): Iterator[P] =
      self(registry, goal, cache, context).map(f)
  }

  def flatMap(f: P => GenTraversableOnce[P]): IDPSolverStep[S, O, P, C] = new IDPSolverStep[S, O, P, C] {
    override def apply(registry: IdRegistry[S], goal: Goal, cache: IDPCache[P, O], context: C): Iterator[P] =
      self(registry, goal, cache, context).flatMap(f)
  }

  def ++(next: IDPSolverStep[S, O, P, C]): IDPSolverStep[S, O, P, C] = new IDPSolverStep[S, O, P, C] {
    override def apply(registry: IdRegistry[S], goal: Goal, cache: IDPCache[P, O], context: C): Iterator[P] =
      self(registry, goal, cache, context) ++ next(registry, goal, cache, context)
  }
}
