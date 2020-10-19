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

import org.neo4j.configuration.GraphDatabaseSettings
import org.neo4j.graphdb.config.Setting

class CommunityRoleAdministrationCommandAcceptanceTest extends CommunityAdministrationCommandAcceptanceTestBase {

  override def databaseConfig(): Map[Setting[_], Object] = super.databaseConfig() ++ Map(GraphDatabaseSettings.auth_enabled -> java.lang.Boolean.TRUE)

  test("should fail on showing roles from community") {
    assertFailure("SHOW ROLES", "Unsupported administration command: SHOW ROLES")
  }

  test("should fail on showing roles with users from community") {
    assertFailure("SHOW POPULATED ROLES WITH USERS", "Unsupported administration command: SHOW POPULATED ROLES WITH USERS")
  }

  test("should fail on creating role from community") {
    assertFailure("CREATE ROLE foo IF NOT EXISTS", "Unsupported administration command: CREATE ROLE foo IF NOT EXISTS")
    assertFailure("CREATE OR REPLACE ROLE foo", "Unsupported administration command: CREATE OR REPLACE ROLE foo")
  }

  test("should fail on creating role from community with correct error message") {
    assertFailure("CREATE ROLE foo", "Unsupported administration command: CREATE ROLE foo")
    assertFailure("CREATE ROLE $foo", "Unsupported administration command: CREATE ROLE $foo")
  }

  test("should fail on creating role as copy of non-existing role with correct error message") {
    assertFailure("CREATE ROLE foo AS COPY OF bar", "Unsupported administration command: CREATE ROLE foo AS COPY OF bar")
    assertFailure("CREATE ROLE foo AS COPY OF $bar", "Unsupported administration command: CREATE ROLE foo AS COPY OF $bar")
  }

  test("should fail on dropping non-existing role from community") {
    assertFailure("DROP ROLE foo IF EXISTS", "Unsupported administration command: DROP ROLE foo IF EXISTS")
  }

  test("should fail on dropping non-existing role from community with correct error message") {
    assertFailure("DROP ROLE foo", "Unsupported administration command: DROP ROLE foo")
    assertFailure("DROP ROLE $foo", "Unsupported administration command: DROP ROLE $foo")
  }

  test("should fail on granting role to user from community") {
    assertFailure("GRANT ROLE reader TO neo4j", "Unsupported administration command: GRANT ROLE reader TO neo4j")
    assertFailure("GRANT ROLE $r TO $u", "Unsupported administration command: GRANT ROLE $r TO $u")
  }

  test("should fail on revoking non-existing role from user") {
    assertFailure("REVOKE ROLE custom FROM neo4j", "Unsupported administration command: REVOKE ROLE custom FROM neo4j")
    assertFailure("REVOKE ROLE $r FROM $u", "Unsupported administration command: REVOKE ROLE $r FROM $u")
  }
}
