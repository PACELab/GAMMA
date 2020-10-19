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

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;

import java.util.concurrent.atomic.AtomicBoolean;

import org.neo4j.configuration.Config;
import org.neo4j.internal.id.DefaultIdGeneratorFactory;
import org.neo4j.io.fs.EphemeralFileSystemAbstraction;
import org.neo4j.io.layout.DatabaseLayout;
import org.neo4j.io.pagecache.PageCache;
import org.neo4j.io.pagecache.tracing.PageCacheTracer;
import org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer;
import org.neo4j.kernel.impl.store.record.AbstractBaseRecord;
import org.neo4j.kernel.impl.store.record.RecordLoad;
import org.neo4j.logging.NullLogProvider;
import org.neo4j.test.extension.EphemeralNeo4jLayoutExtension;
import org.neo4j.test.extension.Inject;
import org.neo4j.test.extension.pagecache.PageCacheSupportExtension;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.neo4j.index.internal.gbptree.RecoveryCleanupWorkCollector.immediate;
import static org.neo4j.kernel.impl.store.record.RecordLoad.NORMAL;
import static org.neo4j.test.rule.PageCacheConfig.config;

@EphemeralNeo4jLayoutExtension
abstract class RecordStoreConsistentReadTest<R extends AbstractBaseRecord, S extends RecordStore<R>>
{
    // Constants for the contents of the existing record
    protected static final int ID = 1;
    @RegisterExtension
    static PageCacheSupportExtension pageCacheExtension = new PageCacheSupportExtension( config().withInconsistentReads( false ) );
    @Inject
    private EphemeralFileSystemAbstraction fs;
    @Inject
    private DatabaseLayout databaseLayout;

    private AtomicBoolean nextReadIsInconsistent;

    @BeforeEach
    void setUp()
    {
        nextReadIsInconsistent = new AtomicBoolean();
    }

    private NeoStores storeFixture()
    {
        PageCache pageCache = pageCacheExtension.getPageCache( fs,
                config().withInconsistentReads( nextReadIsInconsistent ) );
        StoreFactory factory = new StoreFactory( databaseLayout, Config.defaults(), new DefaultIdGeneratorFactory( fs, immediate() ),
                pageCache, fs, NullLogProvider.getInstance(), PageCacheTracer.NULL );
        NeoStores neoStores = factory.openAllNeoStores( true );
        initialiseStore( neoStores );
        return neoStores;
    }

    protected S initialiseStore( NeoStores neoStores )
    {
        S store = getStore( neoStores );
        store.updateRecord( createExistingRecord( false ), PageCursorTracer.NULL );
        return store;
    }

    protected abstract S getStore( NeoStores neoStores );

    protected abstract R createNullRecord( long id );

    protected abstract R createExistingRecord( boolean light );

    protected abstract R getLight( long id, S store );

    protected abstract void assertRecordsEqual( R actualRecord, R expectedRecord );

    protected R getHeavy( S store, long id )
    {
        R record = store.getRecord( id, store.newRecord(), NORMAL, PageCursorTracer.NULL );
        store.ensureHeavy( record, PageCursorTracer.NULL );
        return record;
    }

    R getForce( S store, int id )
    {
        return store.getRecord( id, store.newRecord(), RecordLoad.FORCE, PageCursorTracer.NULL );
    }

    @Test
    void mustReadExistingRecord()
    {
        try ( NeoStores neoStores = storeFixture() )
        {
            S store = getStore( neoStores );
            R record = getHeavy( store, ID );
            assertRecordsEqual( record, createExistingRecord( false ) );
        }
    }

    @Test
    void mustReadExistingLightRecord()
    {
        try ( NeoStores neoStores = storeFixture() )
        {
            S store = getStore( neoStores );
            R record = getLight( ID, store );
            assertRecordsEqual( record, createExistingRecord( true ) );
        }
    }

    @Test
    void mustForceReadExistingRecord()
    {
        try ( NeoStores neoStores = storeFixture() )
        {
            S store = getStore( neoStores );
            R record = getForce( store, ID );
            assertRecordsEqual( record, createExistingRecord( true ) );
        }
    }

    @Test
    void readingNonExistingRecordMustThrow()
    {
        assertThrows( InvalidRecordException.class, () ->
        {
            try ( NeoStores neoStores = storeFixture() )
            {
                S store = getStore( neoStores );
                getHeavy( store, ID + 1 );
            }
        } );
    }

    @Test
    void forceReadingNonExistingRecordMustReturnEmptyRecordWithThatId()
    {
        try ( NeoStores neoStores = storeFixture() )
        {
            S store = getStore( neoStores );
            R record = getForce( store, ID + 1 );
            R nullRecord = createNullRecord( ID + 1 );
            assertRecordsEqual( record, nullRecord );
        }
    }

    @Test
    void mustRetryInconsistentReads()
    {
        try ( NeoStores neoStores = storeFixture() )
        {
            S store = getStore( neoStores );
            nextReadIsInconsistent.set( true );
            R record = getHeavy( store, ID );
            assertRecordsEqual( record, createExistingRecord( false ) );
        }
    }

    @Test
    void mustRetryInconsistentLightReads()
    {
        try ( NeoStores neoStores = storeFixture() )
        {
            S store = getStore( neoStores );
            nextReadIsInconsistent.set( true );
            R record = getLight( ID, store );
            assertRecordsEqual( record, createExistingRecord( true ) );
        }
    }

    @Test
    void mustRetryInconsistentForcedReads()
    {
        try ( NeoStores neoStores = storeFixture() )
        {
            S store = getStore( neoStores );
            nextReadIsInconsistent.set( true );
            R record = getForce( store, ID );
            assertRecordsEqual( record, createExistingRecord( true ) );
        }
    }
}
