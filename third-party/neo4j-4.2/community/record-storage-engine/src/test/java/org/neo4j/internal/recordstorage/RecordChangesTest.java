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
package org.neo4j.internal.recordstorage;

import org.apache.commons.lang3.mutable.MutableInt;
import org.junit.jupiter.api.Test;

import org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer.NULL;
import static org.neo4j.memory.EmptyMemoryTracker.INSTANCE;

class RecordChangesTest
{
    private final RecordAccess.Loader<Object, Object> loader = new RecordAccess.Loader<>()
    {
        @Override
        public Object newUnused( long o, Object additionalData )
        {
            return o;
        }

        @Override
        public Object load( long o, Object additionalData, PageCursorTracer cursorTracer )
        {
            return o;
        }

        @Override
        public void ensureHeavy( Object o, PageCursorTracer cursorTracer )
        {

        }

        @Override
        public Object copy( Object o )
        {
            return o.toString();
        }
    };

    @Test
    void shouldCountChanges()
    {
        // Given
        RecordChanges<Object, Object> change = new RecordChanges<>( loader, new MutableInt(), INSTANCE );

        // When
        change.getOrLoad( 1, null, NULL ).forChangingData();
        change.getOrLoad( 1, null, NULL ).forChangingData();
        change.getOrLoad( 2, null, NULL ).forChangingData();
        change.getOrLoad( 3, null, NULL ).forReadingData();

        // Then
        assertThat( change.changeSize() ).isEqualTo( 2 );
    }
}
