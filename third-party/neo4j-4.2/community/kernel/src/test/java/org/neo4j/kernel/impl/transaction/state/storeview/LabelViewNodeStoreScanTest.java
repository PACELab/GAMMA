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
package org.neo4j.kernel.impl.transaction.state.storeview;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.function.IntPredicate;

import org.neo4j.collection.PrimitiveLongResourceCollections;
import org.neo4j.collection.PrimitiveLongResourceIterator;
import org.neo4j.internal.helpers.collection.Visitor;
import org.neo4j.internal.index.label.LabelScanStore;
import org.neo4j.internal.index.label.TokenScanReader;
import org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer;
import org.neo4j.lock.LockService;
import org.neo4j.storageengine.api.EntityTokenUpdate;
import org.neo4j.storageengine.api.EntityUpdates;
import org.neo4j.storageengine.api.StubStorageCursors;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.neo4j.memory.EmptyMemoryTracker.INSTANCE;

class LabelViewNodeStoreScanTest
{
    private final StubStorageCursors cursors = new StubStorageCursors();
    private final LabelScanStore labelScanStore = mock( LabelScanStore.class );
    private final TokenScanReader labelScanReader = mock( TokenScanReader.class );
    private final IntPredicate propertyKeyIdFilter = mock( IntPredicate.class );
    private final Visitor<EntityTokenUpdate,Exception> labelUpdateVisitor = mock( Visitor.class );
    private final Visitor<EntityUpdates,Exception> propertyUpdateVisitor = mock( Visitor.class );

    @BeforeEach
    void setUp()
    {
        when( labelScanStore.newReader() ).thenReturn( labelScanReader );
    }

    @Test
    void iterateOverLabeledNodeIds()
    {
        PrimitiveLongResourceIterator labeledNodes = PrimitiveLongResourceCollections.iterator( null, 1, 2, 4, 8 );

        long highId = 15L;
        for ( long i = 0; i < highId; i++ )
        {
            cursors.withNode( i );
        }
        int[] labelIds = new int[]{1, 2};
        when( labelScanReader.entitiesWithAnyOfTokens( eq( labelIds ), any() ) ).thenReturn( labeledNodes );

        LabelViewNodeStoreScan<Exception> storeScan = getLabelScanViewStoreScan( labelIds );
        PrimitiveLongResourceIterator idIterator = storeScan.getEntityIdIterator();

        assertThat( idIterator.next() ).isEqualTo( 1L );
        assertThat( idIterator.next() ).isEqualTo( 2L );
        assertThat( idIterator.next() ).isEqualTo( 4L );
        assertThat( idIterator.next() ).isEqualTo( 8L );
        assertThat( idIterator.hasNext() ).isEqualTo( false );
    }

    private LabelViewNodeStoreScan<Exception> getLabelScanViewStoreScan( int[] labelIds )
    {
        return new LabelViewNodeStoreScan<>( cursors, LockService.NO_LOCK_SERVICE,
                labelScanStore, labelUpdateVisitor, propertyUpdateVisitor, labelIds, propertyKeyIdFilter, PageCursorTracer.NULL, INSTANCE );
    }
}
