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

import org.mockito.Mockito.spy
import org.mockito.Mockito.verify
import org.mockito.Mockito.verifyNoMoreInteractions
import org.neo4j.cypher.internal.compiler.planner.logical.ProjectingSelector
import org.neo4j.cypher.internal.util.test_helpers.CypherFunSuite

import scala.collection.immutable.BitSet

class IDPSolverTest extends CypherFunSuite {

  private val context = ()

  private val nullOrderRequirement = new ExtraRequirement[Null, String]() {
    override def none: Null = null

    override def forResult(plan: String): Null = null

    override def is(requirement: Null): Boolean = true
  }

  test("Solves a small toy problem") {
    val monitor = mock[IDPSolverMonitor]
    val solver = new IDPSolver[Char, Null, String, Unit](
      monitor = monitor,
      generator = stringAppendingSolverStep(),
      projectingSelector = firstLongest,
      maxTableSize = 16,
      extraRequirement = nullOrderRequirement,
      iterationDurationLimit = Int.MaxValue
    )

    val seed = Seq(
      (Set('a'), null) -> "a",
      (Set('b'), null) -> "b",
      (Set('c'), null) -> "c",
      (Set('d'), null) -> "d"
    )

    val solution = solver(seed, Set('a', 'b', 'c', 'd'), context)

    solution.toList should equal(List((Set('a', 'b', 'c', 'd'), null) -> "abcd"))
    verify(monitor).foundPlanAfter(1)
  }

  test("Solves a small toy problem with an extra requirement") {
    val monitor = mock[IDPSolverMonitor]
    val capitalization = Capitalization(true)
    val solver = new IDPSolver[Char, Capitalization, String, Unit](
      monitor = monitor,
      generator = stringAppendingSolverStepWithCapitalization(capitalization),
      projectingSelector = firstLongest,
      maxTableSize = 16,
      extraRequirement = CapitalizationRequirement(capitalization),
      iterationDurationLimit = Int.MaxValue
    )

    val seed = Seq(
      (Set('a'), null) -> "a",
      (Set('b'), null) -> "b",
      (Set('c'), null) -> "c",
      (Set('d'), null) -> "d"
    )

    val solution = solver(seed, Set('a', 'b', 'c', 'd'), context)

    solution.toList should equal(List((Set('a', 'b', 'c', 'd'), capitalization) -> "ABCD"))
    verify(monitor).foundPlanAfter(1)
  }

  test("Compacts table at size limit") {
    var table: IDPTable[String, Null] = null
    val monitor = mock[IDPSolverMonitor]
    val solver = new IDPSolver[Char, Null, String, Unit](
      monitor = monitor,
      generator = stringAppendingSolverStep(),
      projectingSelector = firstLongest,
      tableFactory = (registry: IdRegistry[Char], seed: Seed[Char, Null, String]) => {
        table = spy(IDPTable(registry, seed))
        table
      },
      maxTableSize = 4,
      extraRequirement = nullOrderRequirement,
      iterationDurationLimit = Int.MaxValue
    )

    val seed: Seq[((Set[Char], Null), String)] = Seq(
      (Set('a'), null) -> "a",
      (Set('b'), null) -> "b",
      (Set('c'), null) -> "c",
      (Set('d'), null) -> "d",
      (Set('e'), null) -> "e",
      (Set('f'), null) -> "f",
      (Set('g'), null) -> "g",
      (Set('h'), null) -> "h"
    )

    solver(seed, Set('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'), context)

    verify(monitor).startIteration(1)
    verify(monitor).endIteration(1, 2, 16)
    verify(table).removeAllTracesOf(BitSet(0, 1))
    verify(monitor).startIteration(2)
    verify(monitor).endIteration(2, 2, 14)
    verify(table).removeAllTracesOf(BitSet(2, 8))
    verify(monitor).startIteration(3)
    verify(monitor).endIteration(3, 2, 12)
    verify(table).removeAllTracesOf(BitSet(3, 9))
    verify(monitor).startIteration(4)
    verify(monitor).endIteration(4, 2, 10)
    verify(table).removeAllTracesOf(BitSet(4, 10))
    verify(monitor).startIteration(5)
    verify(monitor).endIteration(5, 2, 8)
    verify(table).removeAllTracesOf(BitSet(5, 11))
    verify(monitor).startIteration(6)
    verify(monitor).endIteration(6, 3, 6)
    verify(table).removeAllTracesOf(BitSet(6, 7, 12))
    verify(monitor).foundPlanAfter(6)
    verifyNoMoreInteractions(monitor)
  }

  case class TestIDPSolverMonitor() extends IDPSolverMonitor {
    var maxStartIteration = 0
    var foundPlanIteration = 0

    override def startIteration(iteration: Int): Unit = maxStartIteration = iteration

    override def foundPlanAfter(iterations: Int): Unit = foundPlanIteration = iterations

    override def endIteration(iteration: Int, depth: Int, tableSize: Int): Unit = {}
  }

  def runTimeLimitedSolver(iterationDuration: Int): Int = {
    var table: IDPTable[String, Null] = null
    val monitor = TestIDPSolverMonitor()
    val solver = new IDPSolver[Char, Null, String, Unit](
      monitor = monitor,
      generator = stringAppendingSolverStep(),
      projectingSelector = firstLongest,
      tableFactory = (registry: IdRegistry[Char], seed: Seed[Char, Null, String]) => {
        table = spy(IDPTable(registry, seed))
        table
      },
      maxTableSize = Int.MaxValue,
      extraRequirement = nullOrderRequirement,
      iterationDurationLimit = iterationDuration
    )

    val seed: Seq[((Set[Char], Null), String)] = ('a'.toInt to 'm'.toInt).foldLeft(Seq.empty[((Set[Char], Null), String)]) { (acc, i) =>
      val c = i.toChar
      acc :+ ((Set(c), null) -> c.toString)
    }
    val result = seed.foldLeft(Seq.empty[Char]) { (acc, t) =>
      acc ++ t._1._1
    }.toSet

    solver(seed, result, context)

    monitor.maxStartIteration should equal(monitor.foundPlanIteration)
    monitor.maxStartIteration
  }

  test("Compacts table at time limit") {
    val shortSolverIterations = runTimeLimitedSolver(10)
    val longSolverIterations = runTimeLimitedSolver(1000)
    shortSolverIterations should be > longSolverIterations
  }

  private object firstLongest extends ProjectingSelector[String] {
    override def apply[X](projector: X => String, input: Iterable[X]): Option[X] = {
      val elements = input.toList.sortBy(x => projector(x))
      if (elements.nonEmpty) Some(elements.maxBy(x => projector(x).length)) else None
    }
  }

  private case class stringAppendingSolverStep[O]() extends IDPSolverStep[Char, O, String, Unit] {
    override def apply(registry: IdRegistry[Char], goal: Goal, table: IDPCache[String, O], context: Unit): Iterator[String] = {
      val goalSize = goal.size
      for (
        leftGoal <- goal.subsets if leftGoal.size <= goalSize;
        (_, lhs) <- table(leftGoal);
        rightGoal = goal &~ leftGoal; // bit set -- operator
        (_, rhs) <- table(rightGoal);
        candidate = lhs ++ rhs if isSorted(candidate)
      )
        yield candidate
    }

    def isSorted(chars: String): Boolean =
      (chars.length <= 1) || 0.to(chars.length - 2).forall(i => chars.charAt(i).toInt + 1 == chars.charAt(i + 1).toInt)
  }

  private case class CapitalizationRequirement(capitalization: Capitalization) extends ExtraRequirement[Capitalization, String] {
    override def none: Capitalization = null

    override def forResult(result: String): Capitalization = if (result.equals(capitalization.normalize(result))) capitalization else none

    override def is(requirement: Capitalization): Boolean = requirement == capitalization
  }

  private case class Capitalization(upper: Boolean) {
    def normalize(string: String): String = if (upper) string.toUpperCase else string.toLowerCase
  }

  private case class stringAppendingSolverStepWithCapitalization(capitalization: Capitalization) extends IDPSolverStep[Char, Capitalization, String, Unit] {
    override def apply(registry: IdRegistry[Char], goal: Goal, table: IDPCache[String, Capitalization], context: Unit): Iterator[String] = {
      stringAppendingSolverStep()(registry, goal, table, context).flatMap { candidate =>
        if (capitalization == null)
          Seq(candidate)
        else
          Seq(candidate, capitalization.normalize(candidate))
      }
    }
  }
}
