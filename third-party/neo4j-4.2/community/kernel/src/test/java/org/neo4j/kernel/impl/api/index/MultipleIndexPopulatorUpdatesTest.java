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
package org.neo4j.kernel.impl.api.index;

import org.junit.jupiter.api.Test;
import org.mockito.Answers;

import java.util.function.IntPredicate;
import java.util.function.Supplier;

import org.neo4j.common.EntityType;
import org.neo4j.common.Subject;
import org.neo4j.internal.helpers.collection.Visitor;
import org.neo4j.internal.schema.IndexDescriptor;
import org.neo4j.internal.schema.IndexPrototype;
import org.neo4j.internal.schema.LabelSchemaDescriptor;
import org.neo4j.internal.schema.SchemaDescriptor;
import org.neo4j.internal.schema.SchemaState;
import org.neo4j.io.pagecache.tracing.PageCacheTracer;
import org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer;
import org.neo4j.kernel.api.exceptions.index.IndexEntryConflictException;
import org.neo4j.kernel.api.exceptions.index.IndexPopulationFailedKernelException;
import org.neo4j.kernel.api.index.IndexPopulator;
import org.neo4j.kernel.api.index.IndexUpdater;
import org.neo4j.kernel.impl.api.index.stats.IndexStatisticsStore;
import org.neo4j.kernel.impl.scheduler.JobSchedulerFactory;
import org.neo4j.kernel.impl.transaction.state.storeview.NeoStoreIndexStoreView;
import org.neo4j.kernel.impl.transaction.state.storeview.NodeStoreScan;
import org.neo4j.kernel.impl.util.Listener;
import org.neo4j.lock.LockService;
import org.neo4j.logging.LogProvider;
import org.neo4j.memory.MemoryTracker;
import org.neo4j.storageengine.api.EntityUpdates;
import org.neo4j.storageengine.api.IndexEntryUpdate;
import org.neo4j.storageengine.api.EntityTokenUpdate;
import org.neo4j.storageengine.api.StorageNodeCursor;
import org.neo4j.storageengine.api.StorageReader;
import org.neo4j.test.InMemoryTokens;
import org.neo4j.values.storable.Values;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.neo4j.common.Subject.ANONYMOUS;
import static org.neo4j.common.Subject.AUTH_DISABLED;
import static org.neo4j.memory.EmptyMemoryTracker.INSTANCE;

class MultipleIndexPopulatorUpdatesTest
{
    private final LogProvider logProvider = mock( LogProvider.class, Answers.RETURNS_MOCKS );

    @Test
    void updateForHigherNodeIgnoredWhenUsingFullNodeStoreScan()
            throws IndexPopulationFailedKernelException, IndexEntryConflictException
    {
        IndexStatisticsStore indexStatisticsStore = mock( IndexStatisticsStore.class );

        StorageReader reader = mock( StorageReader.class );
        when( reader.allocateNodeCursor( any() ) ).thenReturn( mock( StorageNodeCursor.class ) );
        ProcessListenableNeoStoreIndexView
                storeView = new ProcessListenableNeoStoreIndexView( LockService.NO_LOCK_SERVICE, () -> reader );
        InMemoryTokens tokens = new InMemoryTokens();
        MultipleIndexPopulator indexPopulator = new MultipleIndexPopulator(
                storeView, logProvider, EntityType.NODE, mock( SchemaState.class ), indexStatisticsStore,
                JobSchedulerFactory.createInitialisedScheduler(), tokens, PageCacheTracer.NULL, INSTANCE, "", AUTH_DISABLED );

        storeView.setProcessListener( new NodeUpdateProcessListener( indexPopulator ) );

        IndexPopulator populator = createIndexPopulator();
        IndexUpdater indexUpdater = mock( IndexUpdater.class );

        addPopulator( indexPopulator, populator, 1, IndexPrototype.forSchema( SchemaDescriptor.forLabel( 1, 1 ) ) );

        indexPopulator.create( PageCursorTracer.NULL );
        StoreScan<IndexPopulationFailedKernelException> storeScan = indexPopulator.createStoreScan( PageCursorTracer.NULL );
        storeScan.run();

        verify( indexUpdater, never() ).process( any(IndexEntryUpdate.class) );
    }

    private static IndexPopulator createIndexPopulator()
    {
        return mock( IndexPopulator.class );
    }

    private static void addPopulator( MultipleIndexPopulator multipleIndexPopulator,
        IndexPopulator indexPopulator, long indexId, IndexPrototype prototype )
    {
        IndexDescriptor descriptor = prototype.withName( "index_" + indexId ).materialise( indexId );
        addPopulator( multipleIndexPopulator, descriptor, indexPopulator, mock( FlippableIndexProxy.class ), mock( FailedIndexProxyFactory.class ) );
    }

    private static void addPopulator( MultipleIndexPopulator multipleIndexPopulator, IndexDescriptor descriptor,
        IndexPopulator indexPopulator, FlippableIndexProxy flippableIndexProxy, FailedIndexProxyFactory failedIndexProxyFactory )
    {
        multipleIndexPopulator.addPopulator( indexPopulator, descriptor, flippableIndexProxy, failedIndexProxyFactory, "userIndexDescription" );
    }

    private static class NodeUpdateProcessListener implements Listener<StorageNodeCursor>
    {
        private final MultipleIndexPopulator indexPopulator;
        private final LabelSchemaDescriptor index;

        NodeUpdateProcessListener( MultipleIndexPopulator indexPopulator )
        {
            this.indexPopulator = indexPopulator;
            this.index = SchemaDescriptor.forLabel( 1, 1 );
        }

        @Override
        public void receive( StorageNodeCursor node )
        {
            if ( node.entityReference() == 7 )
            {
                indexPopulator.queueConcurrentUpdate( IndexEntryUpdate.change( 8L, index, Values.of( "a" ), Values.of( "b" ) ) );
            }
        }
    }

    private static class ProcessListenableNeoStoreIndexView extends NeoStoreIndexStoreView
    {
        private Listener<StorageNodeCursor> processListener;

        ProcessListenableNeoStoreIndexView( LockService locks, Supplier<StorageReader> storageReaderSupplier )
        {
            super( locks, storageReaderSupplier );
        }

        @Override
        public <FAILURE extends Exception> StoreScan<FAILURE> visitNodes( int[] labelIds,
                IntPredicate propertyKeyIdFilter,
                Visitor<EntityUpdates,FAILURE> propertyUpdatesVisitor,
                Visitor<EntityTokenUpdate,FAILURE> labelUpdateVisitor,
                boolean forceStoreScan, PageCursorTracer cursorTracer, MemoryTracker memoryTracker )
        {

            return new ListenableNodeScanViewNodeStoreScan<>( storageEngine.get(), locks, labelUpdateVisitor,
                propertyUpdatesVisitor, labelIds, propertyKeyIdFilter, processListener, cursorTracer );
        }

        void setProcessListener( Listener<StorageNodeCursor> processListener )
        {
            this.processListener = processListener;
        }
    }

    private static class ListenableNodeScanViewNodeStoreScan<FAILURE extends Exception> extends NodeStoreScan<FAILURE>
    {
        private final Listener<StorageNodeCursor> processListener;

        ListenableNodeScanViewNodeStoreScan( StorageReader storageReader, LockService locks,
                Visitor<EntityTokenUpdate,FAILURE> labelUpdateVisitor,
                Visitor<EntityUpdates,FAILURE> propertyUpdatesVisitor, int[] labelIds,
                IntPredicate propertyKeyIdFilter, Listener<StorageNodeCursor> processListener, PageCursorTracer cursorTracer )
        {
            super( storageReader, locks, labelUpdateVisitor, propertyUpdatesVisitor,
                    labelIds, propertyKeyIdFilter, cursorTracer, INSTANCE );
            this.processListener = processListener;
        }

        @Override
        public boolean process( StorageNodeCursor cursor ) throws FAILURE
        {
            processListener.receive( cursor );
            return super.process( cursor );
        }
    }
}
