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
package org.neo4j.consistency.checking.cache;

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import org.neo4j.consistency.checking.full.CheckStage;
import org.neo4j.consistency.checking.full.Stage;
import org.neo4j.consistency.checking.full.StoreProcessor;
import org.neo4j.consistency.statistics.Counts;
import org.neo4j.io.pagecache.tracing.PageCacheTracer;
import org.neo4j.kernel.impl.store.NeoStores;
import org.neo4j.kernel.impl.store.NodeStore;
import org.neo4j.kernel.impl.store.StoreAccess;
import org.neo4j.kernel.impl.store.record.NodeRecord;
import org.neo4j.kernel.impl.store.record.RecordLoad;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.neo4j.memory.EmptyMemoryTracker.INSTANCE;

class CheckNextRelTaskTest
{
    @Test
    void scanForHighIdOnlyOnceWhenProcessCache()
    {
        NeoStores neoStores = mock( NeoStores.class, Mockito.RETURNS_MOCKS );
        NodeStore nodeStore = mock( NodeStore.class );
        NodeRecord nodeRecord = mock( NodeRecord.class );
        StoreProcessor storeProcessor = mock( StoreProcessor.class );

        when( neoStores.getNodeStore() ).thenReturn( nodeStore );
        long highNodeId = 10L;
        when( nodeStore.getHighId() ).thenReturn( highNodeId );
        when( nodeStore.getRecord( anyLong(), any( NodeRecord.class ), any( RecordLoad.class ), any() ) ).thenReturn( nodeRecord );
        when( nodeStore.newRecord() ).thenReturn( nodeRecord );

        StoreAccess storeAccess = new StoreAccess( neoStores );
        storeAccess.initialize();

        DefaultCacheAccess cacheAccess = new DefaultCacheAccess( DefaultCacheAccess.defaultByteArray( highNodeId, INSTANCE ), Counts.NONE, 1 );
        CacheTask.CheckNextRel cacheTask = new CacheTask.CheckNextRel( Stage.SEQUENTIAL_FORWARD, cacheAccess, storeAccess, storeProcessor,
                PageCacheTracer.NULL );

        cacheAccess.setCacheSlotSizes( CheckStage.Stage5_Check_NextRel.getCacheSlotSizes() );
        cacheTask.processCache();

        verify( nodeStore ).getHighId();
    }
}
