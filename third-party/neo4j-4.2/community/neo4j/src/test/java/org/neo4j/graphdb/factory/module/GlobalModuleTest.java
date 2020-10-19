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
package org.neo4j.graphdb.factory.module;

import org.junit.jupiter.api.Test;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Semaphore;

import org.neo4j.configuration.Config;
import org.neo4j.configuration.GraphDatabaseSettings;
import org.neo4j.graphdb.facade.GraphDatabaseDependencies;
import org.neo4j.kernel.impl.factory.DbmsInfo;
import org.neo4j.kernel.impl.scheduler.BufferingExecutor;
import org.neo4j.scheduler.JobMonitoringParams;
import org.neo4j.test.extension.Inject;
import org.neo4j.test.extension.testdirectory.EphemeralTestDirectoryExtension;
import org.neo4j.test.rule.TestDirectory;

import static java.util.concurrent.TimeUnit.MINUTES;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.neo4j.graphdb.facade.GraphDatabaseDependencies.newDependencies;
import static org.neo4j.scheduler.Group.LOG_ROTATION;

@EphemeralTestDirectoryExtension
class GlobalModuleTest
{
    @Inject
    private TestDirectory testDirectory;

    @Test
    void shouldRunDeferredExecutors() throws Exception
    {

        CountDownLatch latch = new CountDownLatch( 1 );
        Semaphore lock = new Semaphore( 1 );

        BufferingExecutor later = new BufferingExecutor();

        // add our later executor to the external dependencies
        GraphDatabaseDependencies externalDependencies = newDependencies().withDeferredExecutor( later, LOG_ROTATION );

        // Take the lock, we're going to use this to synchronize with tasks that run in the executor
        lock.acquire( 1 );

        // add an increment task to the deferred executor
        later.execute( new JobMonitoringParams( null, null, null ), latch::countDown );
        later.execute( new JobMonitoringParams( null, null, null ), lock::release );

        // if I try and get the lock it should fail because the deferred executor is still waiting for a real executor implementation.
        // n.b. this will take the whole timeout time. So don't set this high, even if it means that this test might get lucky and pass
        assertFalse( lock.tryAcquire( 1, 1, SECONDS ) );
        assertThat( latch.getCount() ).isEqualTo( 1L );

        Config cfg = Config.defaults( GraphDatabaseSettings.neo4j_home, testDirectory.absolutePath().toPath() );
        var globalModule = new GlobalModule( cfg, DbmsInfo.UNKNOWN, externalDependencies );
        try
        {
            assertTrue( lock.tryAcquire( 1, MINUTES ) );
            assertTrue( latch.await( 1, MINUTES ) );
        }
        finally
        {
            globalModule.getJobScheduler().shutdown();
        }
    }
}
