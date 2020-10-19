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
package org.neo4j.consistency.checking.full;

import org.junit.jupiter.api.Test;

import org.neo4j.consistency.checking.CheckDecorator;
import org.neo4j.consistency.checking.RecordCheck;
import org.neo4j.consistency.checking.cache.CacheAccess;
import org.neo4j.consistency.report.ConsistencyReport;
import org.neo4j.internal.helpers.progress.ProgressListener;
import org.neo4j.internal.recordstorage.RecordStorageEngine;
import org.neo4j.io.pagecache.PageCursor;
import org.neo4j.io.pagecache.tracing.PageCacheTracer;
import org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer;
import org.neo4j.kernel.impl.store.InvalidRecordException;
import org.neo4j.kernel.impl.store.NeoStores;
import org.neo4j.kernel.impl.store.RecordStore;
import org.neo4j.kernel.impl.store.record.NodeRecord;
import org.neo4j.kernel.impl.store.record.RecordLoad;
import org.neo4j.test.extension.DbmsExtension;
import org.neo4j.test.extension.Inject;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer.NULL;

@SuppressWarnings( "unchecked" )
@DbmsExtension
class StoreProcessorIT
{
    @Inject
    private RecordStorageEngine storageEngine;

    @Test
    void shouldProcessAllTheRecordsInAStore()
    {
        // given
        RecordStore<NodeRecord> nodeStore = getNeoStores().getNodeStore();
        ConsistencyReport.Reporter reporter = mock( ConsistencyReport.Reporter.class );
        StoreProcessor processor = new StoreProcessor( CheckDecorator.NONE,
                reporter, Stage.SEQUENTIAL_FORWARD, CacheAccess.EMPTY );
        nodeStore.updateRecord( node( 0, false, 0, 0 ), NULL );
        nodeStore.updateRecord( node( 1, false, 0, 0 ), NULL );
        nodeStore.updateRecord( node( 2, false, 0, 0 ), NULL );
        nodeStore.setHighestPossibleIdInUse( 2 );

        // when
        processor.applyFiltered( nodeStore, ProgressListener.NONE, PageCacheTracer.NULL );

        // then
        verify( reporter, times( 3 ) ).forNode( any( NodeRecord.class ), any( RecordCheck.class ), any( PageCursorTracer.class ) );
    }

    @Test
    void shouldStopProcessingRecordsWhenSignalledToStop()
    {
        // given
        ConsistencyReport.Reporter reporter = mock( ConsistencyReport.Reporter.class );
        StoreProcessor processor = new StoreProcessor( CheckDecorator.NONE,
                reporter, Stage.SEQUENTIAL_FORWARD, CacheAccess.EMPTY );
        RecordStore<NodeRecord> nodeStore = new RecordStore.Delegator<>( getNeoStores().getNodeStore() )
        {
            @Override
            public void getRecordByCursor( long id, NodeRecord target, RecordLoad mode, PageCursor cursor ) throws InvalidRecordException
            {
                if ( id == 3 )
                {
                    processor.stop();
                }
                super.getRecordByCursor( id, target, mode, cursor );
            }
        };
        nodeStore.updateRecord( node( 0, false, 0, 0 ), NULL );
        nodeStore.updateRecord( node( 1, false, 0, 0 ), NULL );
        nodeStore.updateRecord( node( 2, false, 0, 0 ), NULL );
        nodeStore.updateRecord( node( 3, false, 0, 0 ), NULL );
        nodeStore.updateRecord( node( 4, false, 0, 0 ), NULL );
        nodeStore.setHighestPossibleIdInUse( 4 );

        // when
        processor.applyFiltered( nodeStore, ProgressListener.NONE, PageCacheTracer.NULL );

        // then
        verify( reporter, times( 3 ) ).forNode( any( NodeRecord.class ), any( RecordCheck.class ), any( PageCursorTracer.class ) );
    }

    private NeoStores getNeoStores()
    {
        return storageEngine.testAccessNeoStores();
    }

    private NodeRecord node( long id, boolean dense, long nextRel, long nextProp )
    {
        return new NodeRecord( id ).initialize( true, nextProp, dense, nextRel, 0 );
    }
}
