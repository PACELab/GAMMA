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
package org.neo4j.consistency.store;

import org.mockito.invocation.InvocationOnMock;
import org.mockito.stubbing.Answer;

import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;

import org.neo4j.consistency.checking.CheckerEngine;
import org.neo4j.consistency.checking.ComparativeRecordChecker;
import org.neo4j.consistency.checking.cache.CacheAccess;
import org.neo4j.consistency.checking.cache.CacheTask;
import org.neo4j.consistency.checking.cache.DefaultCacheAccess;
import org.neo4j.consistency.checking.full.CheckStage;
import org.neo4j.consistency.checking.full.MultiPassStore;
import org.neo4j.consistency.checking.full.Stage;
import org.neo4j.consistency.report.ConsistencyReport;
import org.neo4j.consistency.report.PendingReferenceCheck;
import org.neo4j.consistency.statistics.Counts;
import org.neo4j.internal.helpers.ArrayUtil;
import org.neo4j.internal.helpers.collection.IterableWrapper;
import org.neo4j.internal.helpers.collection.PrefetchingIterator;
import org.neo4j.io.pagecache.tracing.PageCacheTracer;
import org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer;
import org.neo4j.kernel.impl.store.PropertyType;
import org.neo4j.kernel.impl.store.record.AbstractBaseRecord;
import org.neo4j.kernel.impl.store.record.DynamicRecord;
import org.neo4j.kernel.impl.store.record.LabelTokenRecord;
import org.neo4j.kernel.impl.store.record.NeoStoreRecord;
import org.neo4j.kernel.impl.store.record.NodeRecord;
import org.neo4j.kernel.impl.store.record.PropertyKeyTokenRecord;
import org.neo4j.kernel.impl.store.record.PropertyRecord;
import org.neo4j.kernel.impl.store.record.Record;
import org.neo4j.kernel.impl.store.record.RelationshipGroupRecord;
import org.neo4j.kernel.impl.store.record.RelationshipRecord;
import org.neo4j.kernel.impl.store.record.RelationshipTypeTokenRecord;
import org.neo4j.kernel.impl.store.record.SchemaRecord;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.neo4j.consistency.checking.cache.DefaultCacheAccess.defaultByteArray;
import static org.neo4j.internal.helpers.collection.Iterables.resourceIterable;
import static org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer.NULL;
import static org.neo4j.memory.EmptyMemoryTracker.INSTANCE;

public class RecordAccessStub implements RecordAccess
{
    public <RECORD extends AbstractBaseRecord, REPORT extends ConsistencyReport>
    CheckerEngine<RECORD, REPORT> engine( final RECORD record, final REPORT report )
    {
        return new Engine<>( report )
        {
            @Override
            @SuppressWarnings( "unchecked" )
            void checkReference( ComparativeRecordChecker checker, AbstractBaseRecord oldReference, AbstractBaseRecord newReference )
            {
                checker.checkReference( record, newReference, this, RecordAccessStub.this, NULL );
            }
        };
    }

    public <RECORD extends AbstractBaseRecord, REPORT extends ConsistencyReport>
    CheckerEngine<RECORD, REPORT> engine( final RECORD oldRecord, final RECORD newRecord, REPORT report )
    {
        return new Engine<>( report )
        {
            @Override
            @SuppressWarnings( "unchecked" )
            void checkReference( ComparativeRecordChecker checker, AbstractBaseRecord oldReference, AbstractBaseRecord newReference )
            {
                checker.checkReference( newRecord, newReference, this, RecordAccessStub.this, NULL );
            }
        };
    }

    private abstract class Engine<RECORD extends AbstractBaseRecord, REPORT extends ConsistencyReport>
            implements CheckerEngine<RECORD, REPORT>
    {
        private final REPORT report;

        protected Engine( REPORT report )
        {
            this.report = report;
        }

        @Override
        public <REFERRED extends AbstractBaseRecord> void comparativeCheck(
                final RecordReference<REFERRED> other,
                final ComparativeRecordChecker<RECORD, ? super REFERRED, REPORT> checker )
        {
            deferredTasks.add( () ->
            {
                PendingReferenceCheck mock = mock( PendingReferenceCheck.class );
                DeferredReferenceCheck check = new DeferredReferenceCheck( Engine.this, checker );
                doAnswer( check ).when( mock ).checkReference( isNull(), isNull(), isNull() );
                doAnswer( check ).when( mock ).checkReference( any( AbstractBaseRecord.class ),
                                                               any( RecordAccess.class ), any( PageCursorTracer.class ) );
                doAnswer( check ).when( mock ).checkDiffReference( any( AbstractBaseRecord.class ),
                                                                   any( AbstractBaseRecord.class ),
                                                                   any( RecordAccess.class ), any( PageCursorTracer.class ) );
                other.dispatch( mock );
            } );
        }

        @Override
        public REPORT report()
        {
            return report;
        }

        abstract void checkReference( ComparativeRecordChecker checker, AbstractBaseRecord oldReference, AbstractBaseRecord newReference );
    }

    private static class DeferredReferenceCheck implements Answer<Void>
    {
        private final Engine dispatch;
        private final ComparativeRecordChecker checker;

        DeferredReferenceCheck( Engine dispatch, ComparativeRecordChecker checker )
        {
            this.dispatch = dispatch;
            this.checker = checker;
        }

        @Override
        public Void answer( InvocationOnMock invocation )
        {
            Object[] arguments = invocation.getArguments();
            AbstractBaseRecord oldReference = null;
            AbstractBaseRecord newReference;
            if ( arguments.length == 4 )
            {
                oldReference = (AbstractBaseRecord) arguments[0];
                newReference = (AbstractBaseRecord) arguments[1];
            }
            else
            {
                newReference = (AbstractBaseRecord) arguments[0];
            }
            dispatch.checkReference( checker, oldReference, newReference );
            return null;
        }
    }

    private final Queue<Runnable> deferredTasks = new LinkedList<>();

    public void checkDeferred()
    {
        for ( Runnable task; null != (task = deferredTasks.poll()); )
        {
            task.run();
        }
    }

    private final Map<Long, Delta<SchemaRecord>> schemata = new HashMap<>();
    private final Map<Long, Delta<NodeRecord>> nodes = new HashMap<>();
    private final Map<Long, Delta<RelationshipRecord>> relationships = new HashMap<>();
    private final Map<Long, Delta<PropertyRecord>> properties = new HashMap<>();
    private final Map<Long, Delta<DynamicRecord>> strings = new HashMap<>();
    private final Map<Long, Delta<DynamicRecord>> arrays = new HashMap<>();
    private final Map<Long, Delta<RelationshipTypeTokenRecord>> relationshipTypeTokens = new HashMap<>();
    private final Map<Long, Delta<LabelTokenRecord>> labelTokens = new HashMap<>();
    private final Map<Long, Delta<PropertyKeyTokenRecord>> propertyKeyTokens = new HashMap<>();
    private final Map<Long, Delta<DynamicRecord>> relationshipTypeNames = new HashMap<>();
    private final Map<Long, Delta<DynamicRecord>> nodeDynamicLabels = new HashMap<>();
    private final Map<Long, Delta<DynamicRecord>> labelNames = new HashMap<>();
    private final Map<Long, Delta<DynamicRecord>> propertyKeyNames = new HashMap<>();
    private final Map<Long, Delta<RelationshipGroupRecord>> relationshipGroups = new HashMap<>();
    private Delta<NeoStoreRecord> graph;
    private final CacheAccess cacheAccess = new DefaultCacheAccess( defaultByteArray( 1_000, INSTANCE ), Counts.NONE, 1 );
    private final MultiPassStore[] storesToCheck;

    public RecordAccessStub()
    {
        this( Stage.SEQUENTIAL_FORWARD, MultiPassStore.values() );
    }

    public RecordAccessStub( Stage stage, MultiPassStore... storesToCheck )
    {
        this.storesToCheck = storesToCheck;
        if ( stage.getCacheSlotSizes().length > 0 )
        {
            cacheAccess.setCacheSlotSizes( stage.getCacheSlotSizes() );
        }
    }

    public void populateCache()
    {
        CacheTask action = new CacheTask.CacheNextRel( CheckStage.Stage3_NS_NextRel, cacheAccess,
                resourceIterable( new IterableWrapper<>( nodes.values() )
                {
                    @Override
                    protected NodeRecord underlyingObjectToObject( Delta<NodeRecord> node )
                    {
                        return node.newRecord;
                    }
                } ), PageCacheTracer.NULL );
        action.run();
    }

    private static class Delta<R extends AbstractBaseRecord>
    {
        final R oldRecord;
        final R newRecord;

        Delta( R record )
        {
            this.oldRecord = null;
            this.newRecord = record;
        }

        Delta( R oldRecord, R newRecord )
        {
            this.oldRecord = oldRecord;
            this.newRecord = newRecord;
        }
    }

    private enum Version
    {
        PREV
        {
            @Override
            <R extends AbstractBaseRecord> R get( Delta<R> delta )
            {
                return delta.oldRecord == null ? delta.newRecord : delta.oldRecord;
            }
        },
        LATEST
        {
            @Override
            <R extends AbstractBaseRecord> R get( Delta<R> delta )
            {
                return delta.newRecord;
            }
        },
        NEW
        {
            @Override
            <R extends AbstractBaseRecord> R get( Delta<R> delta )
            {
                return delta.oldRecord == null ? null : delta.newRecord;
            }
        };

        abstract <R extends AbstractBaseRecord> R get( Delta<R> delta );
    }

    private static <R extends AbstractBaseRecord> R add( Map<Long, Delta<R>> records, R record )
    {
        records.put( record.getId(), new Delta<>( record ) );
        return record;
    }

    private static <R extends AbstractBaseRecord> void add( Map<Long, Delta<R>> records, R oldRecord, R newRecord )
    {
        records.put( newRecord.getId(), new Delta<>( oldRecord, newRecord ) );
    }

    public SchemaRecord addSchema( SchemaRecord schema )
    {
        return add( schemata, schema);
    }

    public DynamicRecord addString( DynamicRecord string )
    {
        return add( strings, string );
    }

    public DynamicRecord addArray( DynamicRecord array )
    {
        return add( arrays, array );
    }

    public DynamicRecord addNodeDynamicLabels( DynamicRecord array )
    {
        return add( nodeDynamicLabels, array );
    }

    public DynamicRecord addPropertyKeyName( DynamicRecord name )
    {
        return add( propertyKeyNames, name );
    }

    public DynamicRecord addRelationshipTypeName( DynamicRecord name )
    {
        return add( relationshipTypeNames, name );
    }

    public DynamicRecord addLabelName( DynamicRecord name )
    {
        return add( labelNames, name );
    }

    public <R extends AbstractBaseRecord> R add( R record )
    {
        if ( record instanceof NodeRecord )
        {
            add( nodes, (NodeRecord) record );
        }
        else if ( record instanceof RelationshipRecord )
        {
            add( relationships, (RelationshipRecord) record );
        }
        else if ( record instanceof PropertyRecord )
        {
            add( properties, (PropertyRecord) record );
        }
        else if ( record instanceof DynamicRecord )
        {
            DynamicRecord dyn = (DynamicRecord) record;
            if ( dyn.getType() == PropertyType.STRING )
            {
                addString( dyn );
            }
            else if ( dyn.getType() == PropertyType.ARRAY )
            {
                addArray( dyn );
            }
            else
            {
                throw new IllegalArgumentException( "Invalid dynamic record type" );
            }
        }
        else if ( record instanceof RelationshipTypeTokenRecord )
        {
            add( relationshipTypeTokens, (RelationshipTypeTokenRecord) record );
        }
        else if ( record instanceof PropertyKeyTokenRecord )
        {
            add( propertyKeyTokens, (PropertyKeyTokenRecord) record );
        }
        else if ( record instanceof LabelTokenRecord )
        {
            add( labelTokens, (LabelTokenRecord) record );
        }
        else if ( record instanceof NeoStoreRecord )
        {
            this.graph = new Delta<>( (NeoStoreRecord) record );
        }
        else if ( record instanceof RelationshipGroupRecord )
        {
            add( relationshipGroups, (RelationshipGroupRecord) record );
        }
        else if ( record instanceof SchemaRecord )
        {
            addSchema( (SchemaRecord) record );
        }
        else
        {
            throw new IllegalArgumentException( "Invalid record type" );
        }
        return record;
    }

    private <R extends AbstractBaseRecord> DirectRecordReference<R> reference( Map<Long, Delta<R>> records,long id, Version version )
    {
        return new DirectRecordReference<>( record( records, id, version ), this, NULL );
    }

    private static <R extends AbstractBaseRecord> R record( Map<Long, Delta<R>> records, long id,
                                                            Version version )
    {
        Delta<R> delta = records.get( id );
        if ( delta == null )
        {
            if ( version == Version.NEW )
            {
                return null;
            }
            throw new AssertionError( String.format( "Access to record with id=%d not expected.", id ) );
        }
        return version.get( delta );
    }

    @Override
    public RecordReference<SchemaRecord> schema( long id, PageCursorTracer cursorTracer )
    {
        return reference( schemata, id, Version.LATEST );
    }

    @Override
    public RecordReference<NodeRecord> node( long id, PageCursorTracer cursorTracer )
    {
        return reference( nodes, id, Version.LATEST );
    }

    @Override
    public RecordReference<RelationshipRecord> relationship( long id, PageCursorTracer cursorTracer )
    {
        return reference( relationships, id, Version.LATEST );
    }

    @Override
    public RecordReference<PropertyRecord> property( long id, PageCursorTracer cursorTracer )
    {
        return reference( properties, id, Version.LATEST );
    }

    @Override
    public Iterator<PropertyRecord> rawPropertyChain( final long firstId, PageCursorTracer cursorTracer )
    {
        return new PrefetchingIterator<>()
        {
            private long next = firstId;

            @Override
            protected PropertyRecord fetchNextOrNull()
            {
                if ( Record.NO_NEXT_PROPERTY.is( next ) )
                {
                    return null;
                }
                PropertyRecord record = reference( properties, next, Version.LATEST ).record();
                next = record.getNextProp();
                return record;
            }
        };
    }

    @Override
    public RecordReference<RelationshipTypeTokenRecord> relationshipType( int id, PageCursorTracer cursorTracer )
    {
        return reference( relationshipTypeTokens, id, Version.LATEST );
    }

    @Override
    public RecordReference<PropertyKeyTokenRecord> propertyKey( int id, PageCursorTracer cursorTracer )
    {
        return reference( propertyKeyTokens, id, Version.LATEST );
    }

    @Override
    public RecordReference<DynamicRecord> string( long id, PageCursorTracer cursorTracer )
    {
        return reference( strings, id, Version.LATEST );
    }

    @Override
    public RecordReference<DynamicRecord> array( long id, PageCursorTracer cursorTracer )
    {
        return reference( arrays, id, Version.LATEST );
    }

    @Override
    public RecordReference<DynamicRecord> relationshipTypeName( int id, PageCursorTracer cursorTracer )
    {
        return reference( relationshipTypeNames, id, Version.LATEST );
    }

    @Override
    public RecordReference<DynamicRecord> nodeLabels( long id, PageCursorTracer cursorTracer )
    {
        return reference( nodeDynamicLabels, id, Version.LATEST );
    }

    @Override
    public RecordReference<LabelTokenRecord> label( int id, PageCursorTracer cursorTracer )
    {
        return reference( labelTokens, id, Version.LATEST );
    }

    @Override
    public RecordReference<DynamicRecord> labelName( int id, PageCursorTracer cursorTracer )
    {
        return reference( labelNames, id, Version.LATEST );
    }

    @Override
    public RecordReference<DynamicRecord> propertyKeyName( int id, PageCursorTracer cursorTracer )
    {
        return reference( propertyKeyNames, id, Version.LATEST );
    }

    @Override
    public RecordReference<RelationshipGroupRecord> relationshipGroup( long id, PageCursorTracer cursorTracer )
    {
        return reference( relationshipGroups, id, Version.LATEST );
    }

    @Override
    public boolean shouldCheck( long id, MultiPassStore store )
    {
        return ArrayUtil.contains( storesToCheck, store );
    }

    @Override
    public CacheAccess cacheAccess()
    {
        return cacheAccess;
    }
}
