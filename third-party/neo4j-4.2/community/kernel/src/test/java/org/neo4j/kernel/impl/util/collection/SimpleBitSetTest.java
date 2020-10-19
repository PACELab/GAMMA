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
package org.neo4j.kernel.impl.util.collection;

import org.eclipse.collections.api.iterator.IntIterator;
import org.eclipse.collections.api.list.primitive.MutableIntList;
import org.eclipse.collections.impl.factory.primitive.IntLists;
import org.eclipse.collections.impl.list.mutable.primitive.IntArrayList;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SimpleBitSetTest
{

    @Test
    void put()
    {
        // Given
        SimpleBitSet set = new SimpleBitSet( 16 );

        // When
        set.put( 2 );
        set.put( 7 );
        set.put( 15 );

        // Then
        assertFalse( set.contains( 1 ) );
        assertFalse( set.contains( 6 ) );
        assertFalse( set.contains( 14 ) );

        assertTrue( set.contains( 2 ) );
        assertTrue( set.contains( 7 ) );
        assertTrue( set.contains( 15 ) );
    }

    @Test
    void putAndRemove()
    {
        // Given
        SimpleBitSet set = new SimpleBitSet( 16 );

        // When
        set.put( 2 );
        set.put( 7 );
        set.remove( 2 );

        // Then
        assertFalse( set.contains( 1 ) );
        assertFalse( set.contains( 6 ) );
        assertFalse( set.contains( 14 ) );
        assertFalse( set.contains( 2 ) );

        assertTrue( set.contains( 7 ) );
    }

    @Test
    void putOtherBitSet()
    {
        // Given
        SimpleBitSet set = new SimpleBitSet( 16 );
        SimpleBitSet otherSet = new SimpleBitSet( 16 );

        otherSet.put( 4 );
        otherSet.put( 14 );

        set.put( 3 );
        set.put( 4 );

        // When
        set.put( otherSet );

        // Then
        assertFalse( set.contains( 0 ) );
        assertFalse( set.contains( 1 ) );
        assertFalse( set.contains( 15 ) );
        assertFalse( set.contains( 7 ) );

        assertTrue( set.contains( 3 ) );
        assertTrue( set.contains( 4 ) );
        assertTrue( set.contains( 14 ) );
    }

    @Test
    void removeOtherBitSet()
    {
        // Given
        SimpleBitSet set = new SimpleBitSet( 16 );
        SimpleBitSet otherSet = new SimpleBitSet( 16 );

        otherSet.put( 4 );
        otherSet.put( 12 );
        otherSet.put( 14 );

        set.put( 3 );
        set.put( 4 );
        set.put( 12 );

        // When
        set.remove( otherSet );

        // Then
        assertFalse( set.contains( 0 ) );
        assertFalse( set.contains( 1 ) );
        assertFalse( set.contains( 4 ) );
        assertFalse( set.contains( 14 ) );

        assertTrue( set.contains( 3 ) );
    }

    @Test
    void resize()
    {
        // Given
        SimpleBitSet set = new SimpleBitSet( 8 );

        // When
        set.put( 128 );

        // Then
        assertTrue( set.contains( 128 ) );

        assertFalse( set.contains( 126 ));
        assertFalse( set.contains( 129 ));
    }

    @Test
    void shouldAllowIterating()
    {
        // Given
        SimpleBitSet set = new SimpleBitSet( 64 );
        set.put( 4 );
        set.put( 7 );
        set.put( 63 );
        set.put( 78 );

        // When
        IntIterator iterator = set.iterator();
        MutableIntList found = new IntArrayList();
        while ( iterator.hasNext() )
        {
            found.add( iterator.next() );
        }

        // Then
        assertThat( found ).isEqualTo( IntLists.immutable.of( 4, 7, 63, 78 ) );
    }

    @Test
    void checkPointOnUnchangedSetMustDoNothing()
    {
        SimpleBitSet set = new SimpleBitSet( 16 );
        int key = 10;
        set.put( key );
        long checkpoint = 0;
        checkpoint = set.checkPointAndPut( checkpoint, key );
        assertThat( set.checkPointAndPut( checkpoint, key ) ).isEqualTo( checkpoint );
        assertTrue( set.contains( key ) );
    }

    @Test
    void checkPointOnUnchangedSetButWithDifferentKeyMustUpdateSet()
    {
        SimpleBitSet set = new SimpleBitSet( 16 );
        int key = 10;
        set.put( key );
        long checkpoint = 0;
        checkpoint = set.checkPointAndPut( checkpoint, key );
        assertThat( set.checkPointAndPut( checkpoint, key + 1 ) ).isNotEqualTo( checkpoint );
        assertTrue( set.contains( key + 1 ) );
        assertFalse( set.contains( key ) );
    }

    @Test
    void checkPointOnChangedSetMustClearState()
    {
        SimpleBitSet set = new SimpleBitSet( 16 );
        int key = 10;
        set.put( key );
        long checkpoint = 0;
        checkpoint = set.checkPointAndPut( checkpoint, key );
        set.put( key + 1 );
        assertThat( set.checkPointAndPut( checkpoint, key ) ).isNotEqualTo( checkpoint );
        assertTrue( set.contains( key ) );
        assertFalse( set.contains( key + 1 ) );
    }

    @Test
    void checkPointMustBeAbleToExpandCapacity()
    {
        SimpleBitSet set = new SimpleBitSet( 16 );
        int key = 10;
        int key2 = 255;
        set.put( key );
        long checkpoint = 0;
        checkpoint = set.checkPointAndPut( checkpoint, key );
        assertThat( set.checkPointAndPut( checkpoint, key2 ) ).isNotEqualTo( checkpoint );
        assertTrue( set.contains( key2 ) );
        assertFalse( set.contains( key ) );
    }

    @Test
    void modificationsMustTakeWriteLocks()
    {
        // We can observe that a write lock was taken, by seeing that an optimistic read lock was invalidated.
        SimpleBitSet set = new SimpleBitSet( 16 );
        long stamp = set.tryOptimisticRead();

        set.put( 8 );
        assertFalse( set.validate( stamp ) );
        stamp = set.tryOptimisticRead();

        set.put( 8 );
        assertFalse( set.validate( stamp ) );
        stamp = set.tryOptimisticRead();

        SimpleBitSet other = new SimpleBitSet( 16 );
        other.put( 3 );
        set.put( other );
        assertFalse( set.validate( stamp ) );
        stamp = set.tryOptimisticRead();

        set.remove( 3 );
        assertFalse( set.validate( stamp ) );
        stamp = set.tryOptimisticRead();

        set.remove( 3 );
        assertFalse( set.validate( stamp ) );
        stamp = set.tryOptimisticRead();

        other.put( 8 );
        set.remove( other );
        assertFalse( set.validate( stamp ) );
        stamp = set.tryOptimisticRead();

        other.put( 8 );
        set.remove( other );
        assertFalse( set.validate( stamp ) );
    }
}
