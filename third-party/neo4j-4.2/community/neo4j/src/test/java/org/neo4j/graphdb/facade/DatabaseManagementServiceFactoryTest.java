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
package org.neo4j.graphdb.facade;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import org.neo4j.configuration.Config;
import org.neo4j.configuration.GraphDatabaseSettings;
import org.neo4j.graphdb.factory.module.GlobalModule;
import org.neo4j.graphdb.factory.module.edition.CommunityEditionModule;
import org.neo4j.kernel.impl.factory.DbmsInfo;
import org.neo4j.kernel.lifecycle.LifeSupport;
import org.neo4j.kernel.lifecycle.Lifecycle;
import org.neo4j.monitoring.Monitors;
import org.neo4j.test.extension.Inject;
import org.neo4j.test.extension.SkipThreadLeakageGuard;
import org.neo4j.test.extension.testdirectory.EphemeralTestDirectoryExtension;
import org.neo4j.test.rule.TestDirectory;

import static org.apache.commons.lang3.exception.ExceptionUtils.getRootCause;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.RETURNS_MOCKS;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@SkipThreadLeakageGuard
@EphemeralTestDirectoryExtension
class DatabaseManagementServiceFactoryTest
{
    @Inject
    private TestDirectory testDirectory;

    private final ExternalDependencies deps = mock( ExternalDependencies.class, RETURNS_MOCKS );

    @BeforeEach
    void setup()
    {
        when( deps.monitors() ).thenReturn( new Monitors() );
        when( deps.dependencies() ).thenReturn( null );
    }

    @Test
    void shouldThrowAppropriateExceptionIfStartFails()
    {
        Config config = Config.defaults( GraphDatabaseSettings.neo4j_home, testDirectory.absolutePath().toPath() );
        RuntimeException startupError = new RuntimeException();
        DatabaseManagementServiceFactory factory = newFaultyGraphDatabaseFacadeFactory( startupError, null );
        RuntimeException startException =
                assertThrows( RuntimeException.class, () -> factory.build( config, deps ) );
        assertEquals( startupError, getRootCause( startException ) );
    }

    @Test
    void shouldThrowAppropriateExceptionIfBothStartAndShutdownFail()
    {
        Config config = Config.defaults( GraphDatabaseSettings.neo4j_home, testDirectory.absolutePath().toPath() );
        RuntimeException startupError = new RuntimeException();
        RuntimeException shutdownError = new RuntimeException();

        DatabaseManagementServiceFactory factory = newFaultyGraphDatabaseFacadeFactory( startupError, shutdownError );
        RuntimeException initException =
                assertThrows( RuntimeException.class, () -> factory.build( config, deps ) );

        assertTrue( initException.getMessage().startsWith( "Error starting " ) );
        assertEquals( startupError, initException.getCause() );
        assertEquals( shutdownError, initException.getSuppressed()[0].getCause() );
    }

    private DatabaseManagementServiceFactory newFaultyGraphDatabaseFacadeFactory( final RuntimeException startupError,
            RuntimeException shutdownError )
    {
        return new DatabaseManagementServiceFactory( DbmsInfo.UNKNOWN, CommunityEditionModule::new )
        {
            @Override
            protected GlobalModule createGlobalModule( Config config, ExternalDependencies dependencies )
            {
                final LifeSupport lifeMock = mock( LifeSupport.class );
                doThrow( startupError ).when( lifeMock ).start();
                if ( shutdownError != null )
                {
                    doThrow( shutdownError ).when( lifeMock ).shutdown();
                }
                doAnswer( invocation -> invocation.getArgument( 0 ) ).when( lifeMock ).add( any( Lifecycle.class ) );

                return new GlobalModule( config, dbmsInfo, dependencies )
                {
                    @Override
                    public LifeSupport createLife()
                    {
                        return lifeMock;
                    }
                };
            }
        };
    }
}
