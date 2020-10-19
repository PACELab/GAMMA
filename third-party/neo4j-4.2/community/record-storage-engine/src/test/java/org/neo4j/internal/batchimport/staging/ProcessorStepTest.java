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
package org.neo4j.internal.batchimport.staging;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

import org.neo4j.internal.batchimport.Configuration;
import org.neo4j.io.pagecache.tracing.DefaultPageCacheTracer;
import org.neo4j.io.pagecache.tracing.PageCacheTracer;
import org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer;
import org.neo4j.test.rule.OtherThreadRule;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.neo4j.io.pagecache.tracing.PageCacheTracer.NULL;

class ProcessorStepTest
{
    private final OtherThreadRule t2 = new OtherThreadRule();

    @BeforeEach
    void setUp()
    {
        t2.init("processor-step");
    }

    @AfterEach
    void tearDown()
    {
        t2.close();
    }

    @Test
    void shouldUpholdProcessOrderingGuarantee() throws Exception
    {
        // GIVEN
        StageControl control = mock( StageControl.class );
        try ( MyProcessorStep step = new MyProcessorStep( control, 0 ) )
        {
            step.start( Step.ORDER_SEND_DOWNSTREAM );
            step.processors( 4 ); // now at 5

            // WHEN
            int batches = 10;
            for ( int i = 0; i < batches; i++ )
            {
                step.receive( i, i );
            }
            step.endOfUpstream();
            step.awaitCompleted();

            // THEN
            assertEquals( batches, step.nextExpected.get() );
        }
    }

    @Test
    void tracePageCacheAccessOnProcess() throws Exception
    {
        StageControl control = mock( StageControl.class );
        var cacheTracer = new DefaultPageCacheTracer();
        int batches = 10;
        try ( MyProcessorStep step = new MyProcessorStep( control, 0, cacheTracer ) )
        {
            step.start( Step.ORDER_SEND_DOWNSTREAM );

            for ( int i = 0; i < batches; i++ )
            {
                step.receive( i, i );
            }
            step.endOfUpstream();
            step.awaitCompleted();

            assertEquals( batches, step.nextExpected.get() );
        }

        assertThat( cacheTracer.pins() ).isEqualTo( batches );
        assertThat( cacheTracer.unpins() ).isEqualTo( batches );
    }

    @Test
    void shouldHaveTaskQueueSizeEqualToMaxNumberOfProcessors() throws Exception
    {
        // GIVEN
        StageControl control = mock( StageControl.class );
        final CountDownLatch latch = new CountDownLatch( 1 );
        final int processors = 2;
        int maxProcessors = 5;
        Configuration configuration = new Configuration()
        {
            @Override
            public int maxNumberOfProcessors()
            {
                return maxProcessors;
            }
        };
        Future<Void> receiveFuture;
        try ( ProcessorStep<Void> step = new BlockingProcessorStep( control, configuration, processors, latch ) )
        {
            step.start( Step.ORDER_SEND_DOWNSTREAM );
            step.processors( 1 ); // now at 2
            // adding up to max processors should be fine
            for ( int i = 0; i < processors + maxProcessors /* +1 since we allow queueing one more*/; i++ )
            {
                step.receive( i, null );
            }

            // WHEN
            receiveFuture = t2.execute( receive( processors, step ) );
            t2.get().waitUntilThreadState( Thread.State.TIMED_WAITING );
            latch.countDown();

            // THEN
            receiveFuture.get();
        }
    }

    @Test
    void shouldRecycleDoneBatches() throws Exception
    {
        // GIVEN
        StageControl control = mock( StageControl.class );
        try ( MyProcessorStep step = new MyProcessorStep( control, 0 ) )
        {
            step.start( Step.ORDER_SEND_DOWNSTREAM );

            // WHEN
            int batches = 10;
            for ( int i = 0; i < batches; i++ )
            {
                step.receive( i, i );
            }
            step.endOfUpstream();
            step.awaitCompleted();

            // THEN
            verify( control, times( batches ) ).recycle( any() );
        }
    }

    private static class BlockingProcessorStep extends ProcessorStep<Void>
    {
        private final CountDownLatch latch;

        BlockingProcessorStep( StageControl control, Configuration configuration,
                int maxProcessors, CountDownLatch latch )
        {
            super( control, "test", configuration, maxProcessors, NULL );
            this.latch = latch;
        }

        @Override
        protected void process( Void batch, BatchSender sender, PageCursorTracer cursorTracer ) throws Throwable
        {
            latch.await();
        }
    }

    private static class MyProcessorStep extends ProcessorStep<Integer>
    {
        private final AtomicInteger nextExpected = new AtomicInteger();

        private MyProcessorStep( StageControl control, int maxProcessors )
        {
            this( control, maxProcessors, NULL );
        }

        private MyProcessorStep( StageControl control, int maxProcessors, PageCacheTracer pageCacheTracer )
        {
            super( control, "test", Configuration.DEFAULT, maxProcessors, pageCacheTracer );
        }

        @Override
        protected void process( Integer batch, BatchSender sender, PageCursorTracer cursorTracer )
        {
            var pinEvent = cursorTracer.beginPin( false, 1, null );
            pinEvent.hit();
            pinEvent.done();
            nextExpected.incrementAndGet();
        }
    }

    private static Callable<Void> receive( final int processors, final ProcessorStep<Void> step )
    {
        return () ->
        {
            step.receive( processors, null );
            return null;
        };
    }
}
