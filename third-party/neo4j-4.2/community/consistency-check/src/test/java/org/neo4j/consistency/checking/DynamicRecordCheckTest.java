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
package org.neo4j.consistency.checking;

import org.junit.jupiter.api.Test;

import org.neo4j.consistency.report.ConsistencyReport;
import org.neo4j.kernel.impl.store.PropertyType;
import org.neo4j.kernel.impl.store.RecordStore;
import org.neo4j.kernel.impl.store.format.standard.DynamicRecordFormat;
import org.neo4j.kernel.impl.store.record.DynamicRecord;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

public abstract class DynamicRecordCheckTest extends RecordCheckTestBase<DynamicRecord,ConsistencyReport.DynamicConsistencyReport,DynamicRecordCheck>
{
    private final int blockSize;

    DynamicRecordCheckTest( DynamicRecordCheck check, int blockSize )
    {
        super( check, ConsistencyReport.DynamicConsistencyReport.class, new int[0] );
        this.blockSize = blockSize;
    }

    @Test
    public void shouldNotReportAnythingForRecordNotInUse()
    {
        // given
        DynamicRecord property = notInUse( record( 42 ) );

        // when
        ConsistencyReport.DynamicConsistencyReport report = check( property );

        // then
        verifyNoMoreInteractions( report );
    }

    @Test
    public void shouldNotReportAnythingForRecordThatDoesNotReferenceOtherRecords()
    {
        // given
        DynamicRecord property = inUse( fill( record( 42 ), blockSize / 2 ) );

        // when
        ConsistencyReport.DynamicConsistencyReport report = check( property );

        // then
        verifyNoMoreInteractions( report );
    }

    @Test
    public void shouldNotReportAnythingForRecordWithConsistentReferences()
    {
        // given
        DynamicRecord property = inUse( fill( record( 42 ) ) );
        DynamicRecord next = add( inUse( fill( record( 7 ), blockSize / 2 ) ) );
        property.setNextBlock( next.getId() );

        // when
        ConsistencyReport.DynamicConsistencyReport report = check( property );

        // then
        verifyNoMoreInteractions( report );
    }

    @Test
    public void shouldReportNextRecordNotInUse()
    {
        // given
        DynamicRecord property = inUse( fill( record( 42 ) ) );
        DynamicRecord next = add( notInUse( record( 7 ) ) );
        property.setNextBlock( next.getId() );

        // when
        ConsistencyReport.DynamicConsistencyReport report = check( property );

        // then
        verify( report ).nextNotInUse( next );
        verifyNoMoreInteractions( report );
    }

    @Test
    public void shouldReportSelfReferentialNext()
    {
        // given
        DynamicRecord property = add( inUse( fill( record( 42 ) ) ) );
        property.setNextBlock( property.getId() );

        // when
        ConsistencyReport.DynamicConsistencyReport report = check( property );

        // then
        verify( report ).circularReferenceNext( any() );
        verifyNoMoreInteractions( report );
    }

    @Test
    public void shouldReportNonFullRecordWithNextReference()
    {
        // given
        DynamicRecord property = inUse( fill( record( 42 ), blockSize - 1 ) );
        DynamicRecord next = add( inUse( fill( record( 7 ), blockSize / 2 ) ) );
        property.setNextBlock( next.getId() );

        // when
        ConsistencyReport.DynamicConsistencyReport report = check( property );

        // then
        verify( report ).recordNotFullReferencesNext();
        verifyNoMoreInteractions( report );
    }

    @Test
    public void shouldReportEmptyRecord()
    {
        // given
        DynamicRecord property = inUse( record( 42 ) );

        // when
        ConsistencyReport.DynamicConsistencyReport report = check( property );

        // then
        verify( report ).emptyBlock();
        verifyNoMoreInteractions( report );
    }

    @Test
    public void shouldReportRecordWithEmptyNext()
    {
        // given
        DynamicRecord property = inUse( fill( record( 42 ) ) );
        DynamicRecord next = add( inUse( record( 7 ) ) );
        property.setNextBlock( next.getId() );

        // when
        ConsistencyReport.DynamicConsistencyReport report = check( property );

        // then
        verify( report ).emptyNextBlock( next );
        verifyNoMoreInteractions( report );
    }

    @Test
    public void shouldReportCorrectTypeBasedOnProperBitsOnly()
    {
        // given
        DynamicRecord property = inUse( record( 42 ) );
        // Type is 9, which is string, but has an extra bit set at a higher up position
        int type = PropertyType.STRING.intValue();
        type = type | 0b10000000;

        property.setType( type );

        // when
        PropertyType reportedType = property.getType();

        // then
        // The type must be string
        assertEquals( PropertyType.STRING, reportedType );
        // but the type data must be preserved
        assertEquals( type, property.getTypeAsInt() );
    }

    // utilities

    DynamicRecord fill( DynamicRecord record )
    {
        return fill( record, blockSize );
    }

    abstract DynamicRecord fill( DynamicRecord record, int size );

    abstract DynamicRecord record( long id );

    public static RecordStore<DynamicRecord> configureDynamicStore( int blockSize )
    {
        @SuppressWarnings( "unchecked" )
        RecordStore<DynamicRecord> mock = mock( RecordStore.class );
        when( mock.getRecordSize() ).thenReturn( blockSize + DynamicRecordFormat.RECORD_HEADER_SIZE );
        when( mock.getRecordDataSize() ).thenReturn( blockSize );
        return mock;
    }
}
