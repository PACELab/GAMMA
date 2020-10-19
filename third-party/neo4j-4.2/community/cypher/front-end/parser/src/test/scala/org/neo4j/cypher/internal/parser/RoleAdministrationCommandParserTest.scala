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
package org.neo4j.cypher.internal.parser

import org.neo4j.cypher.internal.ast
import org.neo4j.cypher.internal.ast.UnaliasedReturnItem
import org.neo4j.cypher.internal.expressions.Equals
import org.neo4j.cypher.internal.util.InputPosition

class RoleAdministrationCommandParserTest extends AdministrationCommandParserTestBase {

  //  Showing roles

  test("SHOW ROLES") {
    yields(ast.ShowRoles(withUsers = false, showAll = true, None, None, None))
  }

  test("CATALOG SHOW ALL ROLES") {
    yields(ast.ShowRoles(withUsers = false, showAll = true, None, None, None))
  }

  test("CATALOG SHOW ALL ROLES YIELD role") {
    yields(ast.ShowRoles(withUsers = false, showAll = true, Some(ast.Return(ast.ReturnItems(includeExisting = false, List(UnaliasedReturnItem(varFor("role"), "role")_))_)_), None, None))
  }

  test("CATALOG SHOW ALL ROLES WHERE role='PUBLIC'") {
    yields(ast.ShowRoles(withUsers = false, showAll = true, None, Some(ast.Where(Equals(varFor("role"), literalString("PUBLIC"))_)_), None))
  }

  test("SHOW ALL ROLES YIELD role RETURN role") {
    failsToParse
  }

  test("SHOW POPULATED ROLES YIELD role WHERE role='PUBLIC' RETURN role") {
    failsToParse
  }

  test("CATALOG SHOW POPULATED ROLES") {
    yields(ast.ShowRoles(withUsers = false, showAll = false, None, None, None))
  }

  test("SHOW ROLES WITH USERS") {
    yields(ast.ShowRoles(withUsers = true, showAll = true, None, None, None))
  }

  test("CATALOG SHOW ALL ROLES WITH USERS") {
    yields(ast.ShowRoles(withUsers = true, showAll = true, None, None, None))
  }

  test("SHOW POPULATED ROLES WITH USERS") {
    yields(ast.ShowRoles(withUsers = true, showAll = false, None, None, None))
  }

  test("CATALOG SHOW ROLE") {
    failsToParse
  }

  test("SHOW ALL ROLE") {
    failsToParse
  }

  test("SHOW POPULATED ROLE") {
    failsToParse
  }

  test("SHOW ROLE role") {
    failsToParse
  }

  test("SHOW ROLE WITH USERS") {
    failsToParse
  }

  test("CATALOG SHOW ROLES WITH USER") {
    failsToParse
  }

  test("SHOW ROLE WITH USER") {
    failsToParse
  }

  test("SHOW ALL ROLE WITH USERS") {
    failsToParse
  }

  test("SHOW ALL ROLES WITH USER") {
    failsToParse
  }

  test("SHOW ALL ROLE WITH USER") {
    failsToParse
  }

  test("CATALOG SHOW POPULATED ROLE WITH USERS") {
    failsToParse
  }

  test("CATALOG SHOW POPULATED ROLES WITH USER") {
    failsToParse
  }

  test("CATALOG SHOW POPULATED ROLE WITH USER") {
    failsToParse
  }

  test("CATALOG SHOW ROLES WITH USER user") {
    failsToParse
  }

  //  Creating role

  test("CREATE ROLE foo") {
    yields(ast.CreateRole(literal("foo"), None, ast.IfExistsThrowError))
  }

  test("CREATE ROLE $foo") {
    yields(ast.CreateRole(param("foo"), None, ast.IfExistsThrowError))
  }

  test("CATALOG CREATE ROLE `foo`") {
    yields(ast.CreateRole(literal("foo"), None, ast.IfExistsThrowError))
  }

  test("CREATE ROLE ``") {
    yields(ast.CreateRole(literal(""), None, ast.IfExistsThrowError))
  }

  test("CREATE ROLE foo AS COPY OF bar") {
    yields(ast.CreateRole(literal("foo"), Some(literal("bar")), ast.IfExistsThrowError))
  }

  test("CREATE ROLE foo AS COPY OF $bar") {
    yields(ast.CreateRole(literal("foo"), Some(param("bar")), ast.IfExistsThrowError))
  }

  test("CREATE ROLE foo AS COPY OF ``") {
    yields(ast.CreateRole(literal("foo"), Some(literal("")), ast.IfExistsThrowError))
  }

  test("CREATE ROLE `` AS COPY OF bar") {
    yields(ast.CreateRole(literal(""), Some(literal("bar")), ast.IfExistsThrowError))
  }

  test("CREATE ROLE foo IF NOT EXISTS") {
    yields(ast.CreateRole(literal("foo"), None, ast.IfExistsDoNothing))
  }

  test("CREATE ROLE foo IF NOT EXISTS AS COPY OF bar") {
    yields(ast.CreateRole(literal("foo"), Some(literal("bar")), ast.IfExistsDoNothing))
  }

  test("CREATE OR REPLACE ROLE foo") {
    yields(ast.CreateRole(literal("foo"), None, ast.IfExistsReplace))
  }

  test("CREATE OR REPLACE ROLE foo AS COPY OF bar") {
    yields(ast.CreateRole(literal("foo"), Some(literal("bar")), ast.IfExistsReplace))
  }

  test("CREATE OR REPLACE ROLE foo IF NOT EXISTS") {
    yields(ast.CreateRole(literal("foo"), None, ast.IfExistsInvalidSyntax))
  }

  test("CREATE OR REPLACE ROLE foo IF NOT EXISTS AS COPY OF bar") {
    yields(ast.CreateRole(literal("foo"), Some(literal("bar")), ast.IfExistsInvalidSyntax))
  }

  test("CATALOG CREATE ROLE \"foo\"") {
    failsToParse
  }

  test("CREATE ROLE f%o") {
    failsToParse
  }

  test("CREATE ROLE  IF NOT EXISTS") {
    failsToParse
  }

  test("CREATE ROLE foo IF EXISTS") {
    failsToParse
  }

  test("CREATE OR REPLACE ROLE ") {
    failsToParse
  }

  test("CREATE ROLE foo AS COPY OF") {
    failsToParse
  }

  test("CREATE ROLE foo IF NOT EXISTS AS COPY OF") {
    failsToParse
  }

  test("CREATE OR REPLACE ROLE foo AS COPY OF") {
    failsToParse
  }

  //  Dropping role

  test("DROP ROLE foo") {
    yields(ast.DropRole(literal("foo"), ifExists = false))
  }

  test("DROP ROLE $foo") {
    yields(ast.DropRole(param("foo"), ifExists = false))
  }

  test("DROP ROLE ``") {
    yields(ast.DropRole(literal(""), ifExists = false))
  }

  test("DROP ROLE foo IF EXISTS") {
    yields(ast.DropRole(literal("foo"), ifExists = true))
  }

  test("DROP ROLE `` IF EXISTS") {
    yields(ast.DropRole(literal(""), ifExists = true))
  }

  test("DROP ROLE ") {
    failsToParse
  }

  test("DROP ROLE  IF EXISTS") {
    failsToParse
  }

  test("DROP ROLE foo IF NOT EXISTS") {
    failsToParse
  }

  //  Granting/revoking roles to/from users

  private type grantOrRevokeRoleFunc = (Seq[String], Seq[String]) => InputPosition => ast.Statement

  private def grantRole(r: Seq[String], u: Seq[String]): InputPosition => ast.Statement = ast.GrantRolesToUsers(r.map(Left(_)), u.map(Left(_)))

  private def revokeRole(r: Seq[String], u: Seq[String]): InputPosition => ast.Statement = ast.RevokeRolesFromUsers(r.map(Left(_)), u.map(Left(_)))

  Seq("ROLE", "ROLES").foreach {
    roleKeyword =>

      Seq(
        ("GRANT", "TO", grantRole: grantOrRevokeRoleFunc),
        ("REVOKE", "FROM", revokeRole: grantOrRevokeRoleFunc)
      ).foreach {
        case (command: String, preposition: String, func: grantOrRevokeRoleFunc) =>

          test(s"$command $roleKeyword foo $preposition abc") {
            yields(func(Seq("foo"), Seq("abc")))
          }

          test(s"CATALOG $command $roleKeyword foo $preposition abc") {
            yields(func(Seq("foo"), Seq("abc")))
          }

          test(s"$command $roleKeyword foo, bar $preposition abc") {
            yields(func(Seq("foo", "bar"), Seq("abc")))
          }

          test(s"$command $roleKeyword foo $preposition abc, def") {
            yields(func(Seq("foo"), Seq("abc", "def")))
          }

          test(s"$command $roleKeyword foo,bla,roo $preposition bar, baz,abc,  def") {
            yields(func(Seq("foo", "bla", "roo"), Seq("bar", "baz", "abc", "def")))
          }

          test(s"$command $roleKeyword `fo:o` $preposition bar") {
            yields(func(Seq("fo:o"), Seq("bar")))
          }

          test(s"$command $roleKeyword foo $preposition `b:ar`") {
            yields(func(Seq("foo"), Seq("b:ar")))
          }

          test(s"$command $roleKeyword `$$f00`,bar $preposition abc,`$$a&c`") {
            yields(func(Seq("$f00", "bar"), Seq("abc", "$a&c")))
          }

          // Should fail to parse if not following the pattern $command $roleKeyword role(s) $preposition user(s)

          test(s"$command $roleKeyword") {
            failsToParse
          }

          test(s"$command $roleKeyword foo") {
            failsToParse
          }

          test(s"$command $roleKeyword foo $preposition") {
            failsToParse
          }

          test(s"$command $roleKeyword $preposition abc") {
            failsToParse
          }

          // Should fail to parse when invalid user or role name

          test(s"$command $roleKeyword fo:o $preposition bar") {
            failsToParse
          }

          test(s"$command $roleKeyword foo $preposition b:ar") {
            failsToParse
          }
      }

      // Should fail to parse when mixing TO and FROM

      test(s"GRANT $roleKeyword foo FROM abc") {
        failsToParse
      }

      test(s"REVOKE $roleKeyword foo TO abc") {
        failsToParse
      }

      // ROLES TO USER only have GRANT and REVOKE and not DENY
      test( s"DENY $roleKeyword foo TO abc") {
        failsToParse
      }
  }

  test("GRANT ROLE $a TO $x") {
    yields(ast.GrantRolesToUsers(Seq(param("a")), Seq(param("x"))))
  }

  test("REVOKE ROLE $a FROM $x") {
    yields(ast.RevokeRolesFromUsers(Seq(param("a")), Seq(param("x"))))
  }

  test("GRANT ROLES a, $b, $c TO $x, y, z") {
    yields(ast.GrantRolesToUsers(Seq(literal("a"), param("b"), param("c")), Seq(param("x"), literal("y"), literal("z"))))
  }

  test("REVOKE ROLES a, $b, $c FROM $x, y, z") {
    yields(ast.RevokeRolesFromUsers(Seq(literal("a"), param("b"), param("c")), Seq(param("x"), literal("y"), literal("z"))))
  }
}
