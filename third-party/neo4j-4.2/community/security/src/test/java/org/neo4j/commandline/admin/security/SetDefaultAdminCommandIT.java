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
package org.neo4j.commandline.admin.security;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import picocli.CommandLine;

import java.io.File;
import java.io.PrintStream;

import org.neo4j.cli.ExecutionContext;
import org.neo4j.configuration.GraphDatabaseSettings;
import org.neo4j.io.fs.EphemeralFileSystemAbstraction;
import org.neo4j.io.fs.FileSystemAbstraction;
import org.neo4j.logging.NullLogProvider;
import org.neo4j.server.security.auth.FileUserRepository;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

class SetDefaultAdminCommandIT
{
    private FileSystemAbstraction fileSystem = new EphemeralFileSystemAbstraction();
    private File confDir;
    private File homeDir;
    private PrintStream out;
    private PrintStream err;

    @BeforeEach
    void setup()
    {
        File graphDir = new File( GraphDatabaseSettings.DEFAULT_DATABASE_NAME );
        confDir = new File( graphDir, "conf" );
        homeDir = new File( graphDir, "home" );
        out = mock( PrintStream.class );
        err = mock( PrintStream.class );
    }

    @Test
    void shouldSetDefaultAdmin() throws Throwable
    {
        execute( "jane" );
        assertAdminIniFile( "jane" );

        verify( out ).println( "default admin user set to 'jane'" );
    }

    @Test
    void shouldOverwrite() throws Throwable
    {
        execute( "jane" );
        assertAdminIniFile( "jane" );
        execute( "janette" );
        assertAdminIniFile( "janette" );

        verify( out ).println( "default admin user set to 'jane'" );
        verify( out ).println( "default admin user set to 'janette'" );
    }

    private void assertAdminIniFile( String username ) throws Throwable
    {
        File adminIniFile = new File( new File( new File( homeDir, "data" ), "dbms" ), SetDefaultAdminCommand.ADMIN_INI );
        Assertions.assertTrue( fileSystem.fileExists( adminIniFile ) );
        FileUserRepository userRepository = new FileUserRepository( fileSystem, adminIniFile,
                NullLogProvider.getInstance() );
        userRepository.start();
        assertThat( userRepository.getAllUsernames() ).contains( username );
        userRepository.stop();
        userRepository.shutdown();
    }

    private void execute( String username )
    {
        final var command = new SetDefaultAdminCommand( new ExecutionContext( homeDir.toPath(), confDir.toPath(), out, err, fileSystem ) );
        CommandLine.populateCommand( command, username );
        command.execute();
    }
}
