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
package org.neo4j.kernel.impl.locking.community;

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import org.neo4j.configuration.Config;
import org.neo4j.lock.LockTracer;
import org.neo4j.lock.ResourceTypes;
import org.neo4j.time.Clocks;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LockManagerImplTest
{
    @Test
    void shouldAllowGetReadWriteLocks()
    {
        // given
        LockResource node1 = new LockResource( ResourceTypes.NODE, 2 );
        LockResource node2 = new LockResource( ResourceTypes.NODE, 2 );
        LockTransaction lockTransaction = new LockTransaction();
        LockManagerImpl lockManager = createLockManager();

        // expect
        assertTrue( lockManager.getReadLock( LockTracer.NONE, node1, lockTransaction ) );
        assertTrue( lockManager.getReadLock( LockTracer.NONE, node2, lockTransaction ) );
        assertTrue( lockManager.getWriteLock( LockTracer.NONE, node2, lockTransaction ) );

        lockManager.releaseReadLock( node1, lockTransaction );
        lockManager.releaseReadLock( node2, lockTransaction );
        lockManager.releaseWriteLock( node2, lockTransaction );

        int lockCount = countLocks( lockManager );
        assertEquals( 0, lockCount );
    }

    @Test
    void shouldNotBePossibleReleaseNotExistingLock()
    {
        // given
        LockResource node1 = new LockResource( ResourceTypes.NODE, 3 );
        LockTransaction lockTransaction = new LockTransaction();
        LockManagerImpl lockManager = createLockManager();

        var e = assertThrows( LockNotFoundException.class, () -> lockManager.releaseReadLock( node1, lockTransaction ) );
        assertThat( e.getMessage() ).startsWith( "Lock not found for: " );
    }

    @Test
    void shouldCleanupNotUsedLocks()
    {
        // given
        LockResource node = new LockResource( ResourceTypes.NODE, 3 );
        LockTransaction lockTransaction = new LockTransaction();
        LockManagerImpl lockManager = createLockManager();
        lockManager.getWriteLock( LockTracer.NONE, node, lockTransaction );

        // expect
        assertTrue( lockManager.tryReadLock( node, lockTransaction ) );
        assertEquals( 1, countLocks( lockManager ) );

        // and when
        lockManager.releaseWriteLock( node, lockTransaction );

        // expect to see one old reader
        assertEquals( 1, countLocks( lockManager ) );

        // and when
        lockManager.releaseReadLock( node, lockTransaction );

        // no more locks left
        assertEquals( 0, countLocks( lockManager ) );
    }

    @Test
    void shouldReleaseNotAcquiredLocks()
    {

        // given
        LockResource node = new LockResource( ResourceTypes.NODE, 3 );
        LockTransaction lockTransaction = new LockTransaction();
        RWLock rwLock = Mockito.mock( RWLock.class );
        LockManagerImpl lockManager = new MockedLockLockManager( new RagManager(), rwLock );

        // expect
        lockManager.tryReadLock( node, lockTransaction );

        // during client close any of the attempts to get read/write lock can be scheduled as last one
        // in that case lock will hot have marks, readers, writers anymore and optimistically created lock
        // need to be removed from global map resource map
        assertEquals( 0, countLocks( lockManager ) );
    }

    private static LockManagerImpl createLockManager()
    {
        return new LockManagerImpl( new RagManager(), Config.defaults(), Clocks.nanoClock() );
    }

    private static int countLocks( LockManagerImpl lockManager )
    {
        final int[] counter = new int[1];
        lockManager.accept( element ->
        {
            counter[0]++;
            return false;
        } );
        return counter[0];
    }

    private static class MockedLockLockManager extends LockManagerImpl
    {

        private final RWLock lock;

        MockedLockLockManager( RagManager ragManager, RWLock lock )
        {
            super( ragManager, Config.defaults(), Clocks.nanoClock() );
            this.lock = lock;
        }

        @Override
        protected RWLock createLock( LockResource resource )
        {
            return lock;
        }
    }
}
