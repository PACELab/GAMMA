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
package org.neo4j.cypher.internal.ast.factory.neo4j

import org.neo4j.cypher.internal.parser.javacc.Cypher
import org.neo4j.cypher.internal.parser.javacc.CypherCharStream
import org.neo4j.cypher.internal.util.OpenCypherExceptionFactory
import org.neo4j.cypher.internal.util.OpenCypherExceptionFactory.SyntaxException
import org.scalatest.Assertions
import org.scalatest.Matchers

import scala.util.Failure
import scala.util.Success
import scala.util.Try

abstract class ParsingTestBase() extends Assertions with Matchers {

  protected def assertSameAST(query: String) = {
    withClue(query+System.lineSeparator()) {
      val parboiledParser = new org.neo4j.cypher.internal.parser.CypherParser()
      val exceptionFactory = new OpenCypherExceptionFactory(None)
      val parboiledAST = Try(parboiledParser.parse(query, exceptionFactory, None))

      val parser = new Cypher(new Neo4jASTFactory(query), new Neo4jASTExceptionFactory(exceptionFactory), new CypherCharStream(query))
      val javaccAST = Try(parser.Statements().get(0))

      (javaccAST, parboiledAST) match {
        case (Failure(javaccEx: SyntaxException), Failure(parboiledEx: SyntaxException)) =>
          withClue(Seq(javaccEx, parboiledEx).mkString("", "\n\n", "\n\n")) {
            javaccEx.pos shouldBe parboiledEx.pos
          }
        case (Failure(javaccEx), Success(_)) =>
          throw javaccEx
        case _ =>
          javaccAST shouldBe parboiledAST
      }
    }
  }
}
