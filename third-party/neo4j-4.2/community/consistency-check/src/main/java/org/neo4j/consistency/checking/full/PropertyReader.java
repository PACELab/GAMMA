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
package org.neo4j.consistency.checking.full;

import org.eclipse.collections.api.set.primitive.MutableLongSet;
import org.eclipse.collections.impl.set.mutable.primitive.LongHashSet;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.neo4j.internal.helpers.collection.Visitor;
import org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer;
import org.neo4j.kernel.impl.store.NodeStore;
import org.neo4j.kernel.impl.store.PropertyStore;
import org.neo4j.kernel.impl.store.StoreAccess;
import org.neo4j.kernel.impl.store.record.NodeRecord;
import org.neo4j.kernel.impl.store.record.PropertyBlock;
import org.neo4j.kernel.impl.store.record.PropertyRecord;
import org.neo4j.kernel.impl.store.record.Record;
import org.neo4j.storageengine.api.NodePropertyAccessor;
import org.neo4j.values.storable.Value;
import org.neo4j.values.storable.Values;

import static org.neo4j.kernel.impl.store.record.RecordLoad.FORCE;

class PropertyReader implements NodePropertyAccessor
{
    private final PropertyStore propertyStore;
    private final NodeStore nodeStore;

    PropertyReader( StoreAccess storeAccess )
    {
        this.propertyStore = storeAccess.getRawNeoStores().getPropertyStore();
        this.nodeStore = storeAccess.getRawNeoStores().getNodeStore();
    }

    Collection<PropertyRecord> getPropertyRecordChain( long firstPropertyRecordId, PageCursorTracer cursorTracer ) throws CircularPropertyRecordChainException
    {
        List<PropertyRecord> records = new ArrayList<>();
        visitPropertyRecordChain( firstPropertyRecordId, record ->
        {
            records.add( record );
            return false; // please continue
        }, cursorTracer );
        return records;
    }

    private boolean visitPropertyRecordChain( long firstPropertyRecordId, Visitor<PropertyRecord,RuntimeException> visitor, PageCursorTracer cursorTracer )
            throws CircularPropertyRecordChainException
    {
        if ( Record.NO_NEXT_PROPERTY.is( firstPropertyRecordId ) )
        {
            return false;
        }

        MutableLongSet visitedPropertyRecordIds = new LongHashSet( 8 );
        visitedPropertyRecordIds.add( firstPropertyRecordId );
        long nextProp = firstPropertyRecordId;
        while ( !Record.NO_NEXT_PROPERTY.is( nextProp ) )
        {
            PropertyRecord propRecord = propertyStore.getRecord( nextProp, propertyStore.newRecord(), FORCE, cursorTracer );
            nextProp = propRecord.getNextProp();
            if ( !Record.NO_NEXT_PROPERTY.is( nextProp ) && !visitedPropertyRecordIds.add( nextProp ) )
            {
                throw new CircularPropertyRecordChainException( propRecord );
            }
            if ( visitor.visit( propRecord ) )
            {
                return true;
            }
        }
        return false;
    }

    public Value propertyValue( PropertyBlock block, PageCursorTracer cursorTracer )
    {
        return block.getType().value( block, propertyStore, cursorTracer );
    }

    @Override
    public Value getNodePropertyValue( long nodeId, int propertyKeyId, PageCursorTracer cursorTracer )
    {
        NodeRecord nodeRecord = nodeStore.newRecord();
        if ( nodeStore.getRecord( nodeId, nodeRecord, FORCE, cursorTracer ).inUse() )
        {
            SpecificValueVisitor visitor = new SpecificValueVisitor( propertyKeyId, cursorTracer );
            try
            {
                if ( visitPropertyRecordChain( nodeRecord.getNextProp(), visitor, cursorTracer ) )
                {
                    return visitor.foundPropertyValue;
                }
            }
            catch ( CircularPropertyRecordChainException e )
            {
                // If we discover a circular reference and still haven't found the property then we won't find it.
                // There are other places where this circular reference will be logged as an inconsistency,
                // so simply catch this exception here and let this method return NO_VALUE below.
            }
        }
        return Values.NO_VALUE;
    }

    private class SpecificValueVisitor implements Visitor<PropertyRecord,RuntimeException>
    {
        private final int propertyKeyId;
        private final PageCursorTracer cursorTracer;
        private Value foundPropertyValue;

        SpecificValueVisitor( int propertyKeyId, PageCursorTracer cursorTracer )
        {
            this.propertyKeyId = propertyKeyId;
            this.cursorTracer = cursorTracer;
        }

        @Override
        public boolean visit( PropertyRecord element ) throws RuntimeException
        {
            for ( PropertyBlock block : element )
            {
                if ( block.getKeyIndexId() == propertyKeyId )
                {
                    foundPropertyValue = propertyValue( block, cursorTracer );
                    return true;
                }
            }
            return false;
        }
    }

    static class CircularPropertyRecordChainException extends Exception
    {
        private final PropertyRecord propertyRecord;

        CircularPropertyRecordChainException( PropertyRecord propertyRecord )
        {
            this.propertyRecord = propertyRecord;
        }

        PropertyRecord propertyRecordClosingTheCircle()
        {
            return propertyRecord;
        }
    }
}
