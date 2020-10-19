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
package org.neo4j.cypher

import java.util.concurrent.Callable
import java.util.concurrent.Executors

import org.neo4j.graphdb.Node

class CypherIsolationIntegrationTest extends ExecutionEngineFunSuite {

  val THREADS = 50
  val UPDATES = 100

  test("Should work around read isolation limitation for simple incremental query") {
    // Given
    val n = createNode("x" -> 0L)

    // When
    race("MATCH (n) SET n.x = n.x + 1")

    // Then
    nodeGetProperty(n, "x") should equal(THREADS * UPDATES)
  }

  test("Should work around read isolation limitation using explicit lock") {
    // Given
    val n = createLabeledNode(Map("x" -> 0L), "L")

    val query =
      """MATCH (n:L) WHERE exists(n.x)
        |SET n._LOCK_ = true
        |WITH n, n.x AS x
        |SET n.x = x + 1
        |REMOVE n._LOCK_""".stripMargin


    // When
    race(query)

    // Then
    nodeGetProperty(n, "x") should equal(THREADS * UPDATES)
  }

  test("Should work around read isolation limitations using explicit lock for cached node properties") {
    // Given
    val n = createLabeledNode(Map("x" -> 0L), "L")
    graph.createIndex("L", "x")

    val query =
      """MATCH (n:L) WHERE exists(n.x)
        |SET n._LOCK_ = true
        |WITH n, n.x AS x
        |SET n.x = x + 1
        |REMOVE n._LOCK_""".stripMargin


    // When
    race(query)

    // Then
    nodeGetProperty(n, "x") should equal(THREADS * UPDATES)
  }

  test("Should work around read isolation limitations using explicit lock for cached node properties with map +=") {
    // Given
    val n = createLabeledNode(Map("x" -> 0L), "L")
    graph.createIndex("L", "x")

    val query =
      """MATCH (n:L) WHERE exists(n.x)
        |SET n += {_LOCK_: true}
        |WITH n, n.x AS x
        |SET n.x = x + 1
        |REMOVE n._LOCK_""".stripMargin


    // When
    race(query)

    // Then
    nodeGetProperty(n, "x") should equal(THREADS * UPDATES)
  }

  private def race(query: String): Unit = {
    val executor = Executors.newFixedThreadPool(THREADS)

    val futures = (1 to THREADS) map { x =>
      executor.submit(new Callable[Unit] {
        override def call(): Unit = {
          for (x <- 1 to UPDATES) {
            execute(query)
          }
        }})
      }

    try {
      futures.foreach(_.get())
    } finally executor.shutdown()
  }

  private def nodeGetProperty(node: Node, property: String): Long = {
    graph.withTx( tx => {
      tx.getNodeById(node.getId).getProperty(property).asInstanceOf[Long]
    } )
  }

}
