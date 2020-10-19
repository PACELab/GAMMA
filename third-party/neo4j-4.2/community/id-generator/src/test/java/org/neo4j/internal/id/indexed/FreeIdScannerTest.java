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
package org.neo4j.internal.id.indexed;

import org.eclipse.collections.api.list.primitive.MutableLongList;
import org.eclipse.collections.impl.factory.primitive.LongLists;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.LongConsumer;

import org.neo4j.index.internal.gbptree.GBPTree;
import org.neo4j.index.internal.gbptree.GBPTreeBuilder;
import org.neo4j.internal.id.indexed.IndexedIdGenerator.ReservedMarker;
import org.neo4j.io.pagecache.PageCache;
import org.neo4j.io.pagecache.tracing.DefaultPageCacheTracer;
import org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer;
import org.neo4j.test.Barrier;
import org.neo4j.test.OtherThreadExecutor;
import org.neo4j.test.extension.Inject;
import org.neo4j.test.extension.pagecache.PageCacheExtension;
import org.neo4j.test.rule.TestDirectory;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.neo4j.internal.id.indexed.IndexedIdGenerator.NO_MONITOR;
import static org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer.NULL;
import static org.neo4j.test.OtherThreadExecutor.command;

@PageCacheExtension
class FreeIdScannerTest
{
    private static final int IDS_PER_ENTRY = 256;

    @Inject
    PageCache pageCache;

    @Inject
    TestDirectory directory;

    private IdRangeLayout layout;
    private GBPTree<IdRangeKey, IdRange> tree;

    // instantiated in tests
    private AtomicBoolean atLeastOneFreeId;
    private ConcurrentLongQueue cache;
    private RecordingReservedMarker reuser;

    @BeforeEach
    void beforeEach()
    {
        this.layout = new IdRangeLayout( IDS_PER_ENTRY );
        this.tree = new GBPTreeBuilder<>( pageCache, directory.filePath( "file.id" ), layout ).build();
    }

    @AfterEach
    void afterEach() throws Exception
    {
        tree.close();
    }

    @Test
    void shouldNotThinkItsWorthScanningIfNoFreedIdsAndNoOngoingScan()
    {
        // given
        FreeIdScanner scanner = scanner( IDS_PER_ENTRY, 8, 1 );

        // then
        assertFalse( scanner.tryLoadFreeIdsIntoCache( NULL ) );
    }

    @Test
    void shouldThinkItsWorthScanningIfAlreadyHasOngoingScan()
    {
        // given
        int generation = 1;
        FreeIdScanner scanner = scanner( IDS_PER_ENTRY, 256, generation );

        forEachId( generation, range( 0, 300 ) ).accept( ( marker, id ) ->
        {
            marker.markDeleted( id );
            marker.markFree( id );
        } );

        // when
        scanner.tryLoadFreeIdsIntoCache( NULL );
        assertTrue( cache.size() > 0 );
        // take at least one so that scanner wants to load more from the ongoing scan
        assertEquals( 0, cache.takeOrDefault( -1 ) );

        // then
        assertTrue( scanner.tryLoadFreeIdsIntoCache( NULL ) );
    }

    @Test
    void shouldFindMarkAndCacheOneIdFromAnEntry()
    {
        // given
        int generation = 1;
        FreeIdScanner scanner = scanner( IDS_PER_ENTRY, 8, generation );

        forEachId( generation, range( 0, 1 ) ).accept( ( marker, id ) ->
        {
            marker.markDeleted( id );
            marker.markFree( id );
        } );

        // when
        scanner.tryLoadFreeIdsIntoCache( NULL );

        // then
        assertCacheHasIds( range( 0, 1 ) );
    }

    @Test
    void shouldFindMarkAndCacheMultipleIdsFromAnEntry()
    {
        // given
        int generation = 1;
        FreeIdScanner scanner = scanner( IDS_PER_ENTRY, 8, generation );
        Range[] ranges = {range( 0, 2 ), range( 7, 8 )}; // 0, 1, 2, 7

        forEachId( generation, ranges ).accept( ( marker, id ) ->
        {
            marker.markDeleted( id );
            marker.markFree( id );
        } );

        // when
        scanner.tryLoadFreeIdsIntoCache( NULL );

        // then
        assertCacheHasIds( ranges );
    }

    @Test
    void shouldFindMarkAndCacheMultipleIdsFromMultipleEntries()
    {
        // given
        int generation = 1;
        FreeIdScanner scanner = scanner( IDS_PER_ENTRY, 16, generation );
        Range[] ranges = {range( 0, 2 ), range( 167, 175 )}; // 0, 1, 2 in one entry and 67,68,69,70,71,72,73,74 in another entry

        forEachId( generation, ranges ).accept( ( marker, id ) ->
        {
            marker.markDeleted( id );
            marker.markFree( id );
        } );

        // when
        scanner.tryLoadFreeIdsIntoCache( NULL );

        // then
        assertCacheHasIds( ranges );
    }

    @Test
    void shouldNotFindUsedIds()
    {
        // given
        int generation = 1;
        FreeIdScanner scanner = scanner( IDS_PER_ENTRY, 16, generation );

        forEachId( generation, range( 0, 5 ) ).accept( ( marker1, id1 ) ->
        {
            marker1.markDeleted( id1 );
            marker1.markFree( id1 );
        } );
        forEachId( generation, range( 1, 3 ) ).accept( ( marker, id ) ->
        {
            marker.markReserved( id );
            marker.markUsed( id );
        } );

        // when
        scanner.tryLoadFreeIdsIntoCache( NULL );

        // then
        assertCacheHasIds( range( 0, 1 ), range( 3, 5 ) );
    }

    @Test
    void shouldNotFindUnusedButNonReusableIds()
    {
        // given
        int generation = 1;
        FreeIdScanner scanner = scanner( IDS_PER_ENTRY, 16, generation );

        forEachId( generation, range( 0, 5 ) ).accept( ( marker, id ) ->
        {
            marker.markDeleted( id );
            marker.markFree( id );
        } );
        forEachId( generation, range( 1, 3 ) ).accept( IdRangeMarker::markReserved );

        // when
        scanner.tryLoadFreeIdsIntoCache( NULL );

        // then
        assertCacheHasIds( range( 0, 1 ), range( 3, 5 ) );
    }

    @Test
    void shouldOnlyScanUntilCacheIsFull()
    {
        // given
        int generation = 1;
        ConcurrentLongQueue cache = mock( ConcurrentLongQueue.class );
        when( cache.capacity() ).thenReturn( 8 );
        when( cache.size() ).thenReturn( 3 );
        when( cache.offer( anyLong() ) ).thenReturn( true );
        FreeIdScanner scanner = scanner( IDS_PER_ENTRY, cache, generation );

        forEachId( generation, range( 0, 8 ) ).accept( ( marker, id ) ->
        {
            marker.markDeleted( id );
            marker.markFree( id );
        } );
        // cache has capacity of 8 and there are 8 free ids, however cache isn't completely empty

        // when
        scanner.tryLoadFreeIdsIntoCache( NULL );

        // then verify that the cache only got offered 5 ids (capacity:8 - size:3)
        verify( cache, times( 5 ) ).offer( anyLong() );
    }

    @Test
    void shouldContinuePausedScan()
    {
        // given
        int generation = 1;
        FreeIdScanner scanner = scanner( IDS_PER_ENTRY, 8, generation );

        forEachId( generation, range( 0, 8 ), range( 64, 72 ) ).accept( ( marker, id ) ->
        {
            marker.markDeleted( id );
            marker.markFree( id );
        } );

        // when
        scanner.tryLoadFreeIdsIntoCache( NULL );

        // then
        assertCacheHasIds( range( 0, 8 ) );

        // and further when
        scanner.tryLoadFreeIdsIntoCache( NULL );

        // then
        assertCacheHasIds( range( 64, 72 ) );
    }

    @Test
    void shouldContinueFromAPausedEntryIfScanWasPausedInTheMiddleOfIt()
    {
        // given
        int generation = 1;
        FreeIdScanner scanner = scanner( IDS_PER_ENTRY, 8, generation );

        forEachId( generation, range( 0, 4 ), range( 64, 72 ) ).accept( ( marker, id ) ->
        {
            marker.markDeleted( id );
            marker.markFree( id );
        } );

        // when
        scanner.tryLoadFreeIdsIntoCache( NULL );

        // then
        assertCacheHasIds( range( 0, 4 ), range( 64, 68 ) );

        // and further when
        scanner.tryLoadFreeIdsIntoCache( NULL );

        // then
        assertCacheHasIds( range( 68, 72 ) );
    }

    @Test
    void shouldOnlyLetOneThreadAtATimePerformAScan() throws Exception
    {
        // given
        int generation = 1;
        Barrier.Control barrier = new Barrier.Control();
        ConcurrentLongQueue cache = mock( ConcurrentLongQueue.class );
        when( cache.capacity() ).thenReturn( 8 );
        when( cache.offer( anyLong() ) ).thenAnswer( invocationOnMock ->
        {
            barrier.reached();
            return true;
        } );
        FreeIdScanner scanner = scanner( IDS_PER_ENTRY, cache, generation );

        forEachId( generation, range( 0, 2 ) ).accept( ( marker, id ) ->
        {
            marker.markDeleted( id );
            marker.markFree( id );
        } );

        // when
        ExecutorService executorService = Executors.newSingleThreadExecutor();
        Future<?> scanFuture = executorService.submit( () -> scanner.tryLoadFreeIdsIntoCache( NULL ) );
        barrier.await();
        // now it's stuck in trying to offer to the cache

        // then a scan call from another thread should complete but not do anything
        verify( cache, times( 1 ) ).offer( anyLong() ); // <-- the 1 call is from the call which makes the other thread stuck above
        scanner.tryLoadFreeIdsIntoCache( NULL );
        verify( cache, times( 1 ) ).offer( anyLong() );

        // clean up
        barrier.release();
        scanFuture.get();
        executorService.shutdown();

        // and then
        verify( cache ).offer( 0 );
        verify( cache ).offer( 1 );
    }

    @Test
    void shouldDisregardReusabilityMarksOnEntriesWithOldGeneration()
    {
        // given
        int oldGeneration = 1;
        int currentGeneration = 2;
        FreeIdScanner scanner = scanner( IDS_PER_ENTRY, 32, currentGeneration );
        forEachId( oldGeneration, range( 0, 8 ), range( 64, 72 ) ).accept( IdRangeMarker::markDeleted );
        // explicitly set to true because the usage pattern in this test is not quite
        atLeastOneFreeId.set( true );

        // when
        scanner.tryLoadFreeIdsIntoCache( NULL );

        // then
        assertCacheHasIds( range( 0, 8 ), range( 64, 72 ) );
    }

    @Test
    void shouldMarkFoundIdsAsNonReusable()
    {
        // given
        long generation = 1;
        FreeIdScanner scanner = scanner( IDS_PER_ENTRY, 32, generation );

        forEachId( generation, range( 0, 5 ) ).accept( ( marker, id ) ->
        {
            marker.markDeleted( id );
            marker.markFree( id );
        } );

        // when
        scanner.tryLoadFreeIdsIntoCache( NULL );

        // then
        assertArrayEquals( new long[]{0, 1, 2, 3, 4}, reuser.reservedIds.toArray() );
    }

    @Test
    void shouldClearCache()
    {
        // given
        long generation = 1;
        ConcurrentLongQueue cache = new SpmcLongQueue( 32 );
        FreeIdScanner scanner = scanner( IDS_PER_ENTRY, cache, generation );
        forEachId( generation, range( 0, 5 ) ).accept( ( marker, id ) ->
        {
            marker.markDeleted( id );
            marker.markFree( id );
        } );
        scanner.tryLoadFreeIdsIntoCache( NULL );

        // when
        long cacheSizeBeforeClear = cache.size();
        scanner.clearCache( NULL );

        // then
        assertEquals( 5, cacheSizeBeforeClear );
        assertEquals( 0, cache.size() );
        assertEquals( LongLists.immutable.of( 0, 1, 2, 3, 4 ), reuser.unreservedIds );
    }

    @Test
    void shouldNotScanWhenConcurrentClear() throws ExecutionException, InterruptedException
    {
        // given
        long generation = 1;
        ConcurrentLongQueue cache = new SpmcLongQueue( 32 );
        Barrier.Control barrier = new Barrier.Control();
        FreeIdScanner scanner = scanner( IDS_PER_ENTRY, new ControlledConcurrentLongQueue( cache, QueueMethodControl.TAKE, barrier ), generation );
        forEachId( generation, range( 0, 5 ) ).accept( ( marker, id ) ->
        {
            marker.markDeleted( id );
            marker.markFree( id );
        } );

        // when
        try ( OtherThreadExecutor clearThread = new OtherThreadExecutor( "clear" ) )
        {
            // Wait for the clear call
            Future<Object> clear = clearThread.executeDontWait( command( () -> scanner.clearCache( NULL ) ) );
            barrier.awaitUninterruptibly();

            // Attempt trigger a scan
            scanner.tryLoadFreeIdsIntoCache( NULL );

            // Let clear finish
            barrier.release();
            clear.get();
        }

        // then
        assertEquals( 0, cache.size() );
    }

    @Test
    void shouldLetClearCacheWaitForConcurrentScan() throws ExecutionException, InterruptedException, TimeoutException
    {
        // given
        long generation = 1;
        ConcurrentLongQueue cache = new SpmcLongQueue( 32 );
        Barrier.Control barrier = new Barrier.Control();
        FreeIdScanner scanner = scanner( IDS_PER_ENTRY, new ControlledConcurrentLongQueue( cache, QueueMethodControl.OFFER, barrier ), generation );
        forEachId( generation, range( 0, 1 ) ).accept( ( marker, id ) ->
        {
            marker.markDeleted( id );
            marker.markFree( id );
        } );

        // when
        try ( OtherThreadExecutor scanThread = new OtherThreadExecutor( "scan" );
              OtherThreadExecutor clearThread = new OtherThreadExecutor( "clear" ) )
        {
            // Wait for the offer call
            Future<Object> scan = scanThread.executeDontWait( command( () -> scanner.tryLoadFreeIdsIntoCache( NULL ) ) );
            barrier.awaitUninterruptibly();

            // Make sure clear waits for the scan call
            Future<Object> clear = clearThread.executeDontWait( command( () -> scanner.clearCache( NULL ) ) );
            clearThread.waitUntilWaiting();

            // Let the threads finish
            barrier.release();
            scan.get();
            clear.get();
        }

        // then
        assertEquals( 0, cache.size() );
    }

    @Test
    void shouldNotSkipRangeThatIsFoundButNoCacheSpaceLeft()
    {
        // given
        long generation = 1;
        int cacheSize = IDS_PER_ENTRY / 2;
        int halfCacheSize = cacheSize / 2;
        FreeIdScanner scanner = scanner( IDS_PER_ENTRY, cacheSize, generation );
        forEachId( generation, range( 0, IDS_PER_ENTRY * 2 + 4 ) ).accept( ( marker, id ) ->
        {
            marker.markDeleted( id );
            marker.markFree( id );
        } );
        scanner.tryLoadFreeIdsIntoCache( NULL );
        assertCacheHasIdsNonExhaustive( range( 0, halfCacheSize ) );
        scanner.tryLoadFreeIdsIntoCache( NULL );
        assertCacheHasIdsNonExhaustive( range( halfCacheSize, cacheSize ) );

        // when
        scanner.tryLoadFreeIdsIntoCache( NULL );

        // then
        assertCacheHasIds( range( cacheSize, IDS_PER_ENTRY ) );
        scanner.tryLoadFreeIdsIntoCache( NULL );
        assertCacheHasIds( range( IDS_PER_ENTRY, IDS_PER_ENTRY + cacheSize ) );
        scanner.tryLoadFreeIdsIntoCache( NULL );
        assertCacheHasIds( range( IDS_PER_ENTRY + cacheSize, IDS_PER_ENTRY * 2 ) );
        scanner.tryLoadFreeIdsIntoCache( NULL );
        assertCacheHasIds( range( IDS_PER_ENTRY * 2, IDS_PER_ENTRY * 2 + 4 ) );
        assertEquals( -1, cache.takeOrDefault( -1 ) );
    }

    @Test
    void shouldEndCurrentScanInClearCache()
    {
        // given
        long generation = 1;
        int cacheSize = IDS_PER_ENTRY / 2;
        int halfCacheSize = cacheSize / 2;
        FreeIdScanner scanner = scanner( IDS_PER_ENTRY, cacheSize, generation );
        forEachId( generation, range( 0, IDS_PER_ENTRY * 2 + 4 ) ).accept( ( marker, id ) ->
        {
            marker.markDeleted( id );
            marker.markFree( id );
        } );
        scanner.tryLoadFreeIdsIntoCache( NULL ); // loads 0 - cacheSize
        assertCacheHasIdsNonExhaustive( range( 0, halfCacheSize ) ); // takes out 0 - cacheSize/2, which means cacheSize/2 - cacheSize is still in cache
        // simulate marking these ids as used and then delete and free them again so that they can be picked up by the scanner after clearCache
        forEachId( generation, range( 0, halfCacheSize ) ).accept( ( marker, id ) ->
        {
            marker.markUsed( id );
            marker.markDeleted( id );
            marker.markFree( id );
        } );

        // when
        scanner.clearCache( NULL ); // should clear cacheSize/2 - cacheSize
        scanner.tryLoadFreeIdsIntoCache( NULL );
        assertCacheHasIdsNonExhaustive( range( 0, halfCacheSize ) );
        assertCacheHasIdsNonExhaustive( range( halfCacheSize, cacheSize ) );
    }

    @Test
    void tracerPageCacheAccessOnCacheScan()
    {
        long generation = 1;
        int cacheSize = IDS_PER_ENTRY / 2;
        FreeIdScanner scanner = scanner( IDS_PER_ENTRY, cacheSize, generation );
        var pageCacheTracer = new DefaultPageCacheTracer();
        try ( var scannerTracer = pageCacheTracer.createPageCursorTracer( "tracerPageCacheAccessOnCacheScan" ) )
        {
            assertThat( scannerTracer.pins() ).isZero();
            assertThat( scannerTracer.unpins() ).isZero();
            assertThat( scannerTracer.hits() ).isZero();

            atLeastOneFreeId.set( true );
            scanner.tryLoadFreeIdsIntoCache( scannerTracer );

            assertThat( scannerTracer.pins() ).isOne();
            assertThat( scannerTracer.unpins() ).isOne();
            assertThat( scannerTracer.hits() ).isOne();
        }
    }

    @Test
    void tracePageCacheAccessOnCacheClear()
    {
        long generation = 1;
        int cacheSize = IDS_PER_ENTRY / 2;
        FreeIdScanner scanner = scanner( IDS_PER_ENTRY, cacheSize, generation );
        var pageCacheTracer = new DefaultPageCacheTracer();
        try ( var scannerTracer = pageCacheTracer.createPageCursorTracer( "tracePageCacheAccessOnCacheClear" ) )
        {
            assertThat( scannerTracer.pins() ).isZero();
            assertThat( scannerTracer.unpins() ).isZero();
            assertThat( scannerTracer.hits() ).isZero();

            scanner.clearCache( scannerTracer );

            assertThat( scannerTracer.pins() ).isOne();
            assertThat( scannerTracer.unpins() ).isOne();
        }
    }

    private FreeIdScanner scanner( int idsPerEntry, int cacheSize, long generation )
    {
        return scanner( idsPerEntry, new SpmcLongQueue( cacheSize ), generation );
    }

    private FreeIdScanner scanner( int idsPerEntry, ConcurrentLongQueue cache, long generation )
    {
        this.cache = cache;
        this.reuser = new RecordingReservedMarker( tree, generation, new AtomicLong() );
        this.atLeastOneFreeId = new AtomicBoolean();
        return new FreeIdScanner( idsPerEntry, tree, cache, atLeastOneFreeId, reuser, generation, false );
    }

    private void assertCacheHasIdsNonExhaustive( Range... ranges )
    {
        assertCacheHasIds( false, ranges );
    }

    private void assertCacheHasIds( Range... ranges )
    {
        assertCacheHasIds( true, ranges );
    }

    private void assertCacheHasIds( boolean exhaustive, Range... ranges )
    {
        for ( Range range : ranges )
        {
            for ( long id = range.fromId; id < range.toId; id++ )
            {
                assertEquals( id, cache.takeOrDefault( -1 ) );
            }
        }
        if ( exhaustive )
        {
            assertEquals( -1, cache.takeOrDefault( -1 ) );
        }
    }

    private Consumer<BiConsumer<IdRangeMarker, Long>> forEachId( long generation, Range... ranges )
    {
        return handler ->
        {
            try ( IdRangeMarker marker = new IdRangeMarker( IDS_PER_ENTRY, layout, tree.writer( NULL ),
                    mock( Lock.class ), IdRangeMerger.DEFAULT, true, atLeastOneFreeId, generation, new AtomicLong(), false, NO_MONITOR ) )
            {
                for ( Range range : ranges )
                {
                    range.forEach( id -> handler.accept( marker, id ) );
                }
            }
            catch ( IOException e )
            {
                throw new UncheckedIOException( e );
            }
        };
    }

    private class RecordingReservedMarker implements MarkerProvider
    {
        private final MutableLongList reservedIds = LongLists.mutable.empty();
        private final MutableLongList unreservedIds = LongLists.mutable.empty();
        private final GBPTree<IdRangeKey,IdRange> tree;
        private final long generation;
        private final AtomicLong highestWrittenId;

        RecordingReservedMarker( GBPTree<IdRangeKey,IdRange> tree, long generation, AtomicLong highestWrittenId )
        {
            this.tree = tree;
            this.generation = generation;
            this.highestWrittenId = highestWrittenId;
        }

        @Override
        public ReservedMarker getMarker( PageCursorTracer cursorTracer )
        {
            ReservedMarker actual = instantiateRealMarker();
            cursorTracer.beginPin( false, 1, null ).done();
            return new ReservedMarker()
            {
                @Override
                public void markReserved( long id )
                {
                    actual.markReserved( id );
                    reservedIds.add( id );
                }

                @Override
                public void markUnreserved( long id )
                {
                    actual.markUnreserved( id );
                    unreservedIds.add( id );
                }

                @Override
                public void close()
                {
                    actual.close();
                }
            };
        }

        private ReservedMarker instantiateRealMarker()
        {
            try
            {
                Lock lock = new ReentrantLock();
                lock.lock();
                return new IdRangeMarker( IDS_PER_ENTRY, layout, tree.writer( NULL ), lock, new IdRangeMerger( false, NO_MONITOR ), true, atLeastOneFreeId,
                        generation, highestWrittenId, false, NO_MONITOR );
            }
            catch ( IOException e )
            {
                throw new UncheckedIOException( e );
            }
        }
    }

    private static Range range( long fromId, long toId )
    {
        return new Range( fromId, toId );
    }

    private static class Range
    {
        private final long fromId;
        private final long toId;

        Range( long fromId, long toId )
        {
            this.fromId = fromId;
            this.toId = toId;
        }

        void forEach( LongConsumer consumer )
        {
            for ( long id = fromId; id < toId; id++ )
            {
                consumer.accept( id );
            }
        }
    }

    private enum QueueMethodControl
    {
        TAKE,
        OFFER
    }

    private static class ControlledConcurrentLongQueue implements ConcurrentLongQueue
    {
        private final ConcurrentLongQueue actual;
        private final QueueMethodControl method;
        private final Barrier.Control barrier;

        ControlledConcurrentLongQueue( ConcurrentLongQueue actual, QueueMethodControl method, Barrier.Control barrier )
        {
            this.actual = actual;
            this.method = method;
            this.barrier = barrier;
        }

        @Override
        public boolean offer( long v )
        {
            if ( method == QueueMethodControl.OFFER )
            {
                barrier.reached();
            }
            return actual.offer( v );
        }

        @Override
        public long takeOrDefault( long defaultValue )
        {
            if ( method == QueueMethodControl.TAKE )
            {
                barrier.reached();
            }
            return actual.takeOrDefault( defaultValue );
        }

        @Override
        public int capacity()
        {
            return actual.capacity();
        }

        @Override
        public int size()
        {
            return actual.size();
        }

        @Override
        public void clear()
        {
            actual.clear();
        }
    }
}
