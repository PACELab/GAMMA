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
package org.neo4j.kernel.impl.store;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;

import org.neo4j.configuration.Config;
import org.neo4j.dbms.api.DatabaseManagementService;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Transaction;
import org.neo4j.io.fs.EphemeralFileSystemAbstraction;
import org.neo4j.io.fs.FileSystemAbstraction;
import org.neo4j.io.layout.DatabaseLayout;
import org.neo4j.io.pagecache.PageCache;
import org.neo4j.io.pagecache.tracing.PageCacheTracer;
import org.neo4j.test.TestDatabaseManagementServiceBuilder;
import org.neo4j.test.extension.EphemeralNeo4jLayoutExtension;
import org.neo4j.test.extension.Inject;
import org.neo4j.test.extension.pagecache.PageCacheSupportExtension;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.neo4j.configuration.Config.defaults;
import static org.neo4j.configuration.GraphDatabaseSettings.DEFAULT_DATABASE_NAME;
import static org.neo4j.kernel.recovery.Recovery.isRecoveryRequired;
import static org.neo4j.memory.EmptyMemoryTracker.INSTANCE;

@EphemeralNeo4jLayoutExtension
class TestStoreAccess
{
    @RegisterExtension
    static PageCacheSupportExtension pageCacheExtension = new PageCacheSupportExtension();
    @Inject
    private EphemeralFileSystemAbstraction fs;
    @Inject
    private DatabaseLayout databaseLayout;

    @Test
    void openingThroughStoreAccessShouldNotTriggerRecovery() throws Throwable
    {
        try ( EphemeralFileSystemAbstraction snapshot = produceUncleanStore() )
        {
            assertTrue( isUnclean( snapshot ), "Store should be unclean" );

            PageCache pageCache = pageCacheExtension.getPageCache( snapshot );
            new StoreAccess( snapshot, pageCache, databaseLayout, Config.defaults(), PageCacheTracer.NULL ).initialize().close();
            assertTrue( isUnclean( snapshot ), "Store should be unclean" );
        }
    }

    private EphemeralFileSystemAbstraction produceUncleanStore()
    {
        DatabaseManagementService managementService = new TestDatabaseManagementServiceBuilder( databaseLayout )
                .setFileSystem( fs )
                .impermanent()
                .build();
        GraphDatabaseService db = managementService.database( DEFAULT_DATABASE_NAME );
        try ( Transaction tx = db.beginTx() )
        {
            tx.createNode();
            tx.commit();
        }
        EphemeralFileSystemAbstraction snapshot = fs.snapshot();
        managementService.shutdown();
        return snapshot;
    }

    private boolean isUnclean( FileSystemAbstraction fileSystem ) throws Exception
    {
        return isRecoveryRequired( fileSystem, databaseLayout, defaults(), INSTANCE );
    }
}
