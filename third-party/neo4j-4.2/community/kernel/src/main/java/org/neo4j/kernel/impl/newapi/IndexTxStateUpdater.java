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
package org.neo4j.kernel.impl.newapi;

import org.apache.commons.lang3.ArrayUtils;
import org.eclipse.collections.api.map.primitive.MutableIntObjectMap;
import org.eclipse.collections.impl.factory.primitive.IntObjectMaps;

import java.util.Arrays;
import java.util.Collection;

import org.neo4j.internal.kernel.api.NodeCursor;
import org.neo4j.internal.kernel.api.PropertyCursor;
import org.neo4j.internal.schema.IndexDescriptor;
import org.neo4j.internal.schema.SchemaDescriptor;
import org.neo4j.kernel.impl.api.index.IndexingService;
import org.neo4j.memory.MemoryTracker;
import org.neo4j.storageengine.api.StorageReader;
import org.neo4j.values.storable.Value;
import org.neo4j.values.storable.ValueTuple;

import static org.neo4j.common.EntityType.NODE;
import static org.neo4j.kernel.api.StatementConstants.NO_SUCH_PROPERTY_KEY;
import static org.neo4j.values.storable.Values.NO_VALUE;

/**
 * Utility class that performs necessary updates for the transaction state.
 */
public class IndexTxStateUpdater
{
    private final StorageReader storageReader;
    private final Read read;
    private final IndexingService indexingService;

    // We can use the StorageReader directly instead of the SchemaReadOps, because we know that in transactions
    // where this class is needed we will never have index changes.
    public IndexTxStateUpdater( StorageReader storageReader, Read read, IndexingService indexingService )
    {
        this.storageReader = storageReader;
        this.read = read;
        this.indexingService = indexingService;
    }

    // LABEL CHANGES

    public enum LabelChangeType
    {
        ADDED_LABEL,
        REMOVED_LABEL
    }

    /**
     * A label has been changed, figure out what updates are needed to tx state.
     *
     * @param labelId The id of the changed label
     * @param existingPropertyKeyIds all property key ids the node has, sorted by id
     * @param node cursor to the node where the change was applied
     * @param propertyCursor cursor to the properties of node
     * @param changeType The type of change event
     */
    void onLabelChange( int labelId, int[] existingPropertyKeyIds, NodeCursor node, PropertyCursor propertyCursor, LabelChangeType changeType )
    {
        assert noSchemaChangedInTx();

        // Check all indexes of the changed label
        Collection<IndexDescriptor> indexes = storageReader.indexesGetRelated( new long[]{labelId}, existingPropertyKeyIds, NODE );
        if ( !indexes.isEmpty() )
        {
            MutableIntObjectMap<Value> materializedProperties = IntObjectMaps.mutable.empty();
            for ( IndexDescriptor index : indexes )
            {
                MemoryTracker memoryTracker = read.txState().memoryTracker();
                int[] indexPropertyIds = index.schema().getPropertyIds();
                Value[] values = getValueTuple( node, propertyCursor, NO_SUCH_PROPERTY_KEY, NO_VALUE, indexPropertyIds, materializedProperties, memoryTracker );
                ValueTuple valueTuple = ValueTuple.of( values );
                memoryTracker.allocateHeap( valueTuple.getShallowSize() );
                switch ( changeType )
                {
                case ADDED_LABEL:
                    indexingService.validateBeforeCommit( index, values );
                    read.txState().indexDoUpdateEntry( index.schema(), node.nodeReference(), null, valueTuple );
                    break;
                case REMOVED_LABEL:
                    read.txState().indexDoUpdateEntry( index.schema(), node.nodeReference(), valueTuple, null );
                    break;
                default:
                    throw new IllegalStateException( changeType + " is not a supported event" );
                }
            }
        }
    }

    private boolean noSchemaChangedInTx()
    {
        return !(read.txState().hasChanges() && !read.txState().hasDataChanges());
    }

    //PROPERTY CHANGES

    void onPropertyAdd( NodeCursor node, PropertyCursor propertyCursor, long[] labels, int propertyKeyId, int[] existingPropertyKeyIds, Value value )
    {
        assert noSchemaChangedInTx();
        Collection<IndexDescriptor> indexes = storageReader.indexesGetRelated( labels, propertyKeyId, NODE );
        if ( !indexes.isEmpty() )
        {
            MutableIntObjectMap<Value> materializedProperties = IntObjectMaps.mutable.empty();
            NodeSchemaMatcher.onMatchingSchema( indexes.iterator(), propertyKeyId, existingPropertyKeyIds,
                    index ->
                    {
                        MemoryTracker memoryTracker = read.txState().memoryTracker();
                        SchemaDescriptor schema = index.schema();
                        Value[] values = getValueTuple( node, propertyCursor, propertyKeyId, value, schema.getPropertyIds(), materializedProperties,
                                                        memoryTracker );
                        indexingService.validateBeforeCommit( index, values );
                        ValueTuple valueTuple = ValueTuple.of( values );
                        memoryTracker.allocateHeap( valueTuple.getShallowSize() );
                        read.txState().indexDoUpdateEntry( schema, node.nodeReference(), null, valueTuple );
                    } );
        }
    }

    void onPropertyRemove( NodeCursor node, PropertyCursor propertyCursor, long[] labels, int propertyKeyId, int[] existingPropertyKeyIds, Value value )
    {
        assert noSchemaChangedInTx();
        Collection<IndexDescriptor> indexes = storageReader.indexesGetRelated( labels, propertyKeyId, NODE );
        if ( !indexes.isEmpty() )
        {
            MutableIntObjectMap<Value> materializedProperties = IntObjectMaps.mutable.empty();
            NodeSchemaMatcher.onMatchingSchema( indexes.iterator(), propertyKeyId, existingPropertyKeyIds,
                    index ->
                    {
                        MemoryTracker memoryTracker = read.txState().memoryTracker();
                        SchemaDescriptor schema = index.schema();
                        Value[] values = getValueTuple( node, propertyCursor, propertyKeyId, value, schema.getPropertyIds(), materializedProperties,
                                                        memoryTracker );
                        ValueTuple valueTuple = ValueTuple.of( values );
                        memoryTracker.allocateHeap( valueTuple.getShallowSize() );
                        read.txState().indexDoUpdateEntry( schema, node.nodeReference(), valueTuple, null );
                    } );
        }
    }

    void onPropertyChange( NodeCursor node, PropertyCursor propertyCursor, long[] labels, int propertyKeyId, int[] existingPropertyKeyIds,
            Value beforeValue, Value afterValue )
    {
        assert noSchemaChangedInTx();
        Collection<IndexDescriptor> indexes = storageReader.indexesGetRelated( labels, propertyKeyId, NODE );
        if ( !indexes.isEmpty() )
        {
            MutableIntObjectMap<Value> materializedProperties = IntObjectMaps.mutable.empty();
            NodeSchemaMatcher.onMatchingSchema( indexes.iterator(), propertyKeyId, existingPropertyKeyIds,
                    index ->
                    {
                        MemoryTracker memoryTracker = read.txState().memoryTracker();
                        SchemaDescriptor schema = index.schema();
                        int[] propertyIds = schema.getPropertyIds();
                        Value[] valuesAfter =
                                getValueTuple( node, propertyCursor, propertyKeyId, afterValue, propertyIds, materializedProperties, memoryTracker );

                        // The valuesBefore tuple is just like valuesAfter, except is has the afterValue instead of the beforeValue
                        Value[] valuesBefore = Arrays.copyOf( valuesAfter, valuesAfter.length );
                        int k = ArrayUtils.indexOf( propertyIds, propertyKeyId );
                        valuesBefore[k] = beforeValue;

                        indexingService.validateBeforeCommit( index, valuesAfter );
                        ValueTuple valuesTupleBefore = ValueTuple.of( valuesBefore );
                        ValueTuple valuesTupleAfter = ValueTuple.of( valuesAfter );
                        memoryTracker.allocateHeap( valuesTupleBefore.getShallowSize() * 2 ); // They are copies and same shallow size
                        read.txState().indexDoUpdateEntry( schema, node.nodeReference(), valuesTupleBefore, valuesTupleAfter );
                    } );
        }
    }

    private Value[] getValueTuple( NodeCursor node, PropertyCursor propertyCursor, int changedPropertyKeyId, Value changedValue, int[] indexPropertyIds,
            MutableIntObjectMap<Value> materializedValues, MemoryTracker memoryTracker )
    {
        Value[] values = new Value[indexPropertyIds.length];
        int missing = 0;

        // First get whatever values we already have on the stack, like the value change that provoked this update in the first place
        // and already loaded values that we can get from the map of materialized values.
        for ( int k = 0; k < indexPropertyIds.length; k++ )
        {
            values[k] = indexPropertyIds[k] == changedPropertyKeyId ? changedValue : materializedValues.getIfAbsent( indexPropertyIds[k], () -> NO_VALUE );
            if ( values[k] == NO_VALUE )
            {
                missing++;
            }
        }

        // If we couldn't get all values that we wanted we need to load from the node. While we're loading values
        // we'll place those values in the map so that other index updates from this change can just used them.
        if ( missing > 0 )
        {
            node.properties( propertyCursor );
            while ( missing > 0 && propertyCursor.next() )
            {
                int k = ArrayUtils.indexOf( indexPropertyIds, propertyCursor.propertyKey() );
                if ( k >= 0 && values[k] == NO_VALUE )
                {
                    int propertyKeyId = indexPropertyIds[k];
                    boolean thisIsTheChangedProperty = propertyKeyId == changedPropertyKeyId;
                    values[k] = thisIsTheChangedProperty ? changedValue : propertyCursor.propertyValue();
                    if ( !thisIsTheChangedProperty )
                    {
                        materializedValues.put( propertyKeyId, values[k] );
                        memoryTracker.allocateHeap( values[k].estimatedHeapUsage() );
                    }
                    missing--;
                }
            }
        }

        return values;
    }
}
