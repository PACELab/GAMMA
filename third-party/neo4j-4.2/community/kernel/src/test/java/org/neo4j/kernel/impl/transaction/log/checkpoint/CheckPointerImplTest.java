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
package org.neo4j.kernel.impl.transaction.log.checkpoint;

import org.junit.jupiter.api.Test;

import java.io.Flushable;
import java.io.IOException;
import java.time.Duration;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.BooleanSupplier;

import org.neo4j.function.ThrowingConsumer;
import org.neo4j.io.pagecache.IOLimiter;
import org.neo4j.io.pagecache.tracing.PageCacheTracer;
import org.neo4j.kernel.database.DatabaseTracers;
import org.neo4j.kernel.impl.transaction.log.LogPosition;
import org.neo4j.kernel.impl.transaction.log.TransactionAppender;
import org.neo4j.kernel.impl.transaction.log.checkpoint.CheckPointerImpl.ForceOperation;
import org.neo4j.kernel.impl.transaction.log.pruning.LogPruning;
import org.neo4j.kernel.impl.transaction.tracing.DatabaseTracer;
import org.neo4j.kernel.impl.transaction.tracing.LogCheckPointEvent;
import org.neo4j.logging.NullLogProvider;
import org.neo4j.monitoring.DatabaseHealth;
import org.neo4j.monitoring.Health;
import org.neo4j.storageengine.api.TransactionIdStore;
import org.neo4j.util.concurrent.BinaryLatch;

import static java.time.Duration.ofMinutes;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.RETURNS_MOCKS;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.reset;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;
import static org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer.NULL;
import static org.neo4j.test.ThreadTestUtils.forkFuture;

class CheckPointerImplTest
{
    private static final SimpleTriggerInfo INFO = new SimpleTriggerInfo( "Test" );
    public static final Duration TIMEOUT = ofMinutes( 5 );

    private final TransactionIdStore txIdStore = mock( TransactionIdStore.class );
    private final CheckPointThreshold threshold = mock( CheckPointThreshold.class );
    private final ForceOperation forceOperation = mock( ForceOperation.class );
    private final LogPruning logPruning = mock( LogPruning.class );
    private final TransactionAppender appender = mock( TransactionAppender.class );
    private final Health health = mock( DatabaseHealth.class );
    private final DatabaseTracer tracer = mock( DatabaseTracer.class, RETURNS_MOCKS );
    private IOLimiter limiter = mock( IOLimiter.class );

    private final long initialTransactionId = 2L;
    private final long transactionId = 42L;
    private final LogPosition logPosition = new LogPosition( 16L, 233L );

    @Test
    void shouldNotFlushIfItIsNotNeeded() throws Throwable
    {
        // Given
        CheckPointerImpl checkPointing = checkPointer();
        when( threshold.isCheckPointingNeeded( anyLong(), any( TriggerInfo.class ) ) ).thenReturn( false );

        checkPointing.start();

        // When
        long txId = checkPointing.checkPointIfNeeded( INFO );

        // Then
        assertEquals( -1, txId );
        verifyNoInteractions( forceOperation );
        verifyNoInteractions( tracer );
        verifyNoInteractions( appender );
    }

    @Test
    void shouldFlushIfItIsNeeded() throws Throwable
    {
        // Given
        CheckPointerImpl checkPointing = checkPointer();
        when( threshold.isCheckPointingNeeded( anyLong(), eq( INFO ) ) ).thenReturn( true, false );
        mockTxIdStore();

        checkPointing.start();

        // When
        long txId = checkPointing.checkPointIfNeeded( INFO );

        // Then
        assertEquals( transactionId, txId );
        verify( forceOperation ).flushAndForce( limiter, NULL );
        verify( health, times( 2 ) ).assertHealthy( IOException.class );
        verify( appender ).checkPoint( eq( logPosition ), any( LogCheckPointEvent.class ) );
        verify( threshold ).initialize( initialTransactionId );
        verify( threshold ).checkPointHappened( transactionId );
        verify( threshold ).isCheckPointingNeeded( transactionId, INFO );
        verify( logPruning ).pruneLogs( logPosition.getLogVersion() );
        verify( tracer ).beginCheckPoint();
        verifyNoMoreInteractions( forceOperation, health, appender, threshold, tracer );
    }

    @Test
    void shouldForceCheckPointAlways() throws Throwable
    {
        // Given
        CheckPointerImpl checkPointing = checkPointer();
        when( threshold.isCheckPointingNeeded( anyLong(), eq( INFO ) ) ).thenReturn( false );
        mockTxIdStore();

        checkPointing.start();

        // When
        long txId = checkPointing.forceCheckPoint( INFO );

        // Then
        assertEquals( transactionId, txId );
        verify( forceOperation ).flushAndForce( limiter, NULL );
        verify( health, times( 2 ) ).assertHealthy( IOException.class );
        verify( appender ).checkPoint( eq( logPosition ), any( LogCheckPointEvent.class ) );
        verify( threshold ).initialize( initialTransactionId );
        verify( threshold ).checkPointHappened( transactionId );
        verify( threshold, never() ).isCheckPointingNeeded( transactionId, INFO );
        verify( logPruning ).pruneLogs( logPosition.getLogVersion() );
        verifyNoMoreInteractions( forceOperation, health, appender, threshold );
    }

    @Test
    void shouldCheckPointAlwaysWhenThereIsNoRunningCheckPoint() throws Throwable
    {
        // Given
        CheckPointerImpl checkPointing = checkPointer();
        when( threshold.isCheckPointingNeeded( anyLong(), eq( INFO ) ) ).thenReturn( false );
        mockTxIdStore();

        checkPointing.start();

        // When
        long txId = checkPointing.tryCheckPoint( INFO );

        // Then
        assertEquals( transactionId, txId );
        verify( forceOperation ).flushAndForce( limiter, NULL );
        verify( health, times( 2 ) ).assertHealthy( IOException.class );
        verify( appender ).checkPoint( eq( logPosition ), any( LogCheckPointEvent.class ) );
        verify( threshold ).initialize( initialTransactionId );
        verify( threshold ).checkPointHappened( transactionId );
        verify( threshold, never() ).isCheckPointingNeeded( transactionId, INFO );
        verify( logPruning ).pruneLogs( logPosition.getLogVersion() );
        verifyNoMoreInteractions( forceOperation, health, appender, threshold );
    }

    @Test
    void shouldCheckPointNoWaitAlwaysWhenThereIsNoRunningCheckPoint() throws Throwable
    {
        // Given
        CheckPointerImpl checkPointing = checkPointer();
        when( threshold.isCheckPointingNeeded( anyLong(), eq( INFO ) ) ).thenReturn( false );
        mockTxIdStore();

        checkPointing.start();

        // When
        long txId = checkPointing.tryCheckPointNoWait( INFO );

        // Then
        assertEquals( transactionId, txId );
        verify( forceOperation ).flushAndForce( limiter, NULL );
        verify( health, times( 2 ) ).assertHealthy( IOException.class );
        verify( appender ).checkPoint( eq( logPosition ), any( LogCheckPointEvent.class ) );
        verify( threshold ).initialize( initialTransactionId );
        verify( threshold ).checkPointHappened( transactionId );
        verify( threshold, never() ).isCheckPointingNeeded( transactionId, INFO );
        verify( logPruning ).pruneLogs( logPosition.getLogVersion() );
        verifyNoMoreInteractions( forceOperation, health, appender, threshold );
    }

    @Test
    void forceCheckPointShouldWaitTheCurrentCheckPointingToCompleteBeforeRunning() throws Throwable
    {
        // Given
        Lock lock = new ReentrantLock();
        final Lock spyLock = spy( lock );

        doAnswer( invocation ->
        {
            verify( appender ).checkPoint( any( LogPosition.class ), any( LogCheckPointEvent.class ) );
            reset( appender );
            invocation.callRealMethod();
            return null;
        } ).when( spyLock ).unlock();

        final CheckPointerImpl checkPointing = checkPointer( mutex( spyLock ) );
        mockTxIdStore();

        final CountDownLatch startSignal = new CountDownLatch( 2 );
        final CountDownLatch completed = new CountDownLatch( 2 );

        checkPointing.start();

        Thread checkPointerThread = new CheckPointerThread( checkPointing, startSignal, completed );

        Thread forceCheckPointThread = new Thread( () ->
        {
            try
            {
                startSignal.countDown();
                startSignal.await();
                checkPointing.forceCheckPoint( INFO );

                completed.countDown();
            }
            catch ( Throwable e )
            {
                throw new RuntimeException( e );
            }
        } );

        // when
        checkPointerThread.start();
        forceCheckPointThread.start();

        completed.await();

        verify( spyLock, times( 2 ) ).lock();
        verify( spyLock, times( 2 ) ).unlock();
    }

    private StoreCopyCheckPointMutex mutex( Lock lock )
    {
        return new StoreCopyCheckPointMutex( new ReadWriteLock()
        {
            @Override
            public Lock writeLock()
            {
                return lock;
            }

            @Override
            public Lock readLock()
            {
                throw new UnsupportedOperationException();
            }
        } );
    }

    @Test
    void tryCheckPointShouldWaitTheCurrentCheckPointingToCompleteNoRunCheckPointButUseTheTxIdOfTheEarlierRun()
            throws Throwable
    {
        // Given
        Lock lock = mock( Lock.class );
        when( lock.tryLock( anyLong(), any( TimeUnit.class ) ) ).thenReturn( true );
        final CheckPointerImpl checkPointing = checkPointer( mutex( lock ) );
        mockTxIdStore();

        checkPointing.forceCheckPoint( INFO );

        verify( appender ).checkPoint( eq( logPosition ), any( LogCheckPointEvent.class ) );
        reset( appender );

        checkPointing.tryCheckPoint( INFO );

        verifyNoMoreInteractions( appender );
    }

    @Test
    void tryCheckPointNoWaitShouldReturnWhenCheckPointIsAlreadyRunning() throws Throwable
    {
        // Given
        Lock lock = mock( Lock.class );
        when( lock.tryLock() ).thenReturn( false );
        CheckPointerImpl checkPointing = checkPointer( mutex( lock ) );
        mockTxIdStore();

        // When
        long id = checkPointing.tryCheckPointNoWait( INFO );

        // Then
        assertEquals( -1, id );
        verifyNoMoreInteractions( appender );
    }

    @Test
    void mustUseIoLimiterFromFlushing() throws Throwable
    {
        limiter = new IOLimiter()
        {
            @Override
            public long maybeLimitIO( long previousStamp, int recentlyCompletedIOs, Flushable flushable )
            {
                return 42;
            }

            @Override
            public boolean isLimited()
            {
                return true;
            }
        };
        when( threshold.isCheckPointingNeeded( anyLong(), eq( INFO ) ) ).thenReturn( true, false );
        mockTxIdStore();
        CheckPointerImpl checkPointing = checkPointer();

        checkPointing.start();
        checkPointing.checkPointIfNeeded( INFO );

        verify( forceOperation ).flushAndForce( limiter, NULL );
    }

    @Test
    void mustFlushAsFastAsPossibleDuringForceCheckPoint() throws Exception
    {
        AtomicBoolean doneDisablingLimits = new AtomicBoolean();
        limiter = new IOLimiter()
        {
            @Override
            public long maybeLimitIO( long previousStamp, int recentlyCompletedIOs, Flushable flushable )
            {
                return 0;
            }

            @Override
            public void enableLimit()
            {
                doneDisablingLimits.set( true );
            }

            @Override
            public boolean isLimited()
            {
                return doneDisablingLimits.get();
            }
        };
        mockTxIdStore();
        CheckPointerImpl checkPointer = checkPointer();
        checkPointer.forceCheckPoint( new SimpleTriggerInfo( "test" ) );
        assertTrue( doneDisablingLimits.get() );
    }

    @Test
    void mustFlushAsFastAsPossibleDuringTryCheckPoint() throws Exception
    {

        AtomicBoolean doneDisablingLimits = new AtomicBoolean();
        limiter = new IOLimiter()
        {
            @Override
            public long maybeLimitIO( long previousStamp, int recentlyCompletedIOs, Flushable flushable )
            {
                return 0;
            }

            @Override
            public void enableLimit()
            {
                doneDisablingLimits.set( true );
            }

            @Override
            public boolean isLimited()
            {
                return doneDisablingLimits.get();
            }
        };
        mockTxIdStore();
        CheckPointerImpl checkPointer = checkPointer();
        checkPointer.tryCheckPoint( INFO );
        assertTrue( doneDisablingLimits.get() );
    }

    @Test
    void tryCheckPointMustWaitForOnGoingCheckPointsToCompleteAsLongAsTimeoutPredicateIsFalse() throws Exception
    {
        mockTxIdStore();
        CheckPointerImpl checkPointer = checkPointer();
        BinaryLatch arriveFlushAndForce = new BinaryLatch();
        BinaryLatch finishFlushAndForce = new BinaryLatch();

        doAnswer( invocation ->
        {
            arriveFlushAndForce.release();
            finishFlushAndForce.await();
            return null;
        } ).when( forceOperation ).flushAndForce( limiter, NULL );

        Thread forceCheckPointThread = new Thread( () ->
        {
            try
            {
                checkPointer.forceCheckPoint( INFO );
            }
            catch ( Throwable e )
            {
                e.printStackTrace();
                throw new RuntimeException( e );
            }
        } );
        forceCheckPointThread.start();

        arriveFlushAndForce.await(); // Wait for force-thread to arrive in flushAndForce().

        BooleanSupplier predicate = mock( BooleanSupplier.class );
        when( predicate.getAsBoolean() ).thenReturn( false, false, true );
        assertThat( checkPointer.tryCheckPoint( INFO, predicate ) ).isEqualTo( -1L ); // We decided to not wait for the on-going check point to finish.

        finishFlushAndForce.release(); // Let the flushAndForce complete.
        forceCheckPointThread.join();

        assertThat( checkPointer.tryCheckPoint( INFO, predicate ) ).isEqualTo( this.transactionId );
    }

    private void verifyAsyncActionCausesConcurrentFlushingRush(
            ThrowingConsumer<CheckPointerImpl,IOException> asyncAction ) throws Exception
    {
        AtomicLong limitDisableCounter = new AtomicLong();
        AtomicLong observedRushCount = new AtomicLong();
        BinaryLatch backgroundCheckPointStartedLatch = new BinaryLatch();
        BinaryLatch forceCheckPointStartLatch = new BinaryLatch();

        limiter = new IOLimiter()
        {
            @Override
            public long maybeLimitIO( long previousStamp, int recentlyCompletedIOs, Flushable flushable )
            {
                return 0;
            }

            @Override
            public void disableLimit()
            {
                limitDisableCounter.getAndIncrement();
                forceCheckPointStartLatch.release();
            }

            @Override
            public void enableLimit()
            {
                limitDisableCounter.getAndDecrement();
            }

            @Override
            public boolean isLimited()
            {
                return limitDisableCounter.get() != 0;
            }
        };

        mockTxIdStore();
        CheckPointerImpl checkPointer = checkPointer();

        doAnswer( invocation ->
        {
            backgroundCheckPointStartedLatch.release();
            forceCheckPointStartLatch.await();
            long newValue = limitDisableCounter.get();
            observedRushCount.set( newValue );
            return null;
        } ).when( forceOperation ).flushAndForce( limiter, NULL );

        Future<Object> forceCheckPointer = forkFuture( () ->
        {
            backgroundCheckPointStartedLatch.await();
            asyncAction.accept( checkPointer );
            return null;
        } );

        when( threshold.isCheckPointingNeeded( anyLong(), eq( INFO ) ) ).thenReturn( true );
        checkPointer.checkPointIfNeeded( INFO );
        forceCheckPointer.get();
        assertThat( observedRushCount.get() ).isEqualTo( 1L );
    }

    @Test
    void mustRequestFastestPossibleFlushWhenForceCheckPointIsCalledDuringBackgroundCheckPoint()
    {
        assertTimeoutPreemptively( TIMEOUT, () ->
                verifyAsyncActionCausesConcurrentFlushingRush( checkPointer -> checkPointer.forceCheckPoint( new SimpleTriggerInfo( "async" ) ) ) );

    }

    @Test
    void mustRequestFastestPossibleFlushWhenTryCheckPointIsCalledDuringBackgroundCheckPoint()
    {
        assertTimeoutPreemptively( TIMEOUT, () ->
                verifyAsyncActionCausesConcurrentFlushingRush( checkPointer -> checkPointer.tryCheckPoint( new SimpleTriggerInfo( "async" ) ) ) );
    }

    private CheckPointerImpl checkPointer( StoreCopyCheckPointMutex mutex )
    {
        var databaseTracers = mock( DatabaseTracers.class );
        when( databaseTracers.getDatabaseTracer() ).thenReturn( tracer );
        when( databaseTracers.getPageCacheTracer() ).thenReturn( PageCacheTracer.NULL );
        return new CheckPointerImpl( txIdStore, threshold, forceOperation, logPruning, appender, health,
                NullLogProvider.getInstance(), databaseTracers, limiter, mutex );
    }

    private CheckPointerImpl checkPointer()
    {
        return checkPointer( new StoreCopyCheckPointMutex() );
    }

    private void mockTxIdStore()
    {
        long[] triggerCommittedTransaction = {transactionId, logPosition.getLogVersion(), logPosition.getByteOffset()};
        when( txIdStore.getLastClosedTransaction() ).thenReturn( triggerCommittedTransaction );
        when( txIdStore.getLastClosedTransactionId() ).thenReturn( initialTransactionId, transactionId, transactionId );
    }

    private static class CheckPointerThread extends Thread
    {
        private final CheckPointerImpl checkPointing;
        private final CountDownLatch startSignal;
        private final CountDownLatch completed;

        CheckPointerThread( CheckPointerImpl checkPointing, CountDownLatch startSignal, CountDownLatch completed )
        {
            this.checkPointing = checkPointing;
            this.startSignal = startSignal;
            this.completed = completed;
        }

        @Override
        public void run()
        {
            try
            {
                startSignal.countDown();
                startSignal.await();
                checkPointing.forceCheckPoint( INFO );
                completed.countDown();
            }
            catch ( Exception e )
            {
                throw new RuntimeException( e );
            }
        }
    }
}
