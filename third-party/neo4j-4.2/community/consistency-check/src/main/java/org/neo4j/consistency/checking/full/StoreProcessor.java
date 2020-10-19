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

import org.neo4j.consistency.RecordType;
import org.neo4j.consistency.checking.AbstractStoreProcessor;
import org.neo4j.consistency.checking.CheckDecorator;
import org.neo4j.consistency.checking.RecordCheck;
import org.neo4j.consistency.checking.SchemaRecordCheck;
import org.neo4j.consistency.checking.cache.CacheAccess;
import org.neo4j.consistency.checking.full.QueueDistribution.QueueDistributor;
import org.neo4j.consistency.report.ConsistencyReport;
import org.neo4j.consistency.report.ConsistencyReport.DynamicLabelConsistencyReport;
import org.neo4j.consistency.report.ConsistencyReport.RelationshipGroupConsistencyReport;
import org.neo4j.consistency.statistics.Counts;
import org.neo4j.graphdb.ResourceIterable;
import org.neo4j.graphdb.ResourceIterator;
import org.neo4j.internal.helpers.progress.ProgressListener;
import org.neo4j.io.pagecache.tracing.PageCacheTracer;
import org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer;
import org.neo4j.kernel.impl.store.RecordStore;
import org.neo4j.kernel.impl.store.record.AbstractBaseRecord;
import org.neo4j.kernel.impl.store.record.DynamicRecord;
import org.neo4j.kernel.impl.store.record.LabelTokenRecord;
import org.neo4j.kernel.impl.store.record.NodeRecord;
import org.neo4j.kernel.impl.store.record.PropertyKeyTokenRecord;
import org.neo4j.kernel.impl.store.record.PropertyRecord;
import org.neo4j.kernel.impl.store.record.RelationshipGroupRecord;
import org.neo4j.kernel.impl.store.record.RelationshipRecord;
import org.neo4j.kernel.impl.store.record.RelationshipTypeTokenRecord;
import org.neo4j.kernel.impl.store.record.SchemaRecord;

import static org.neo4j.consistency.checking.cache.DefaultCacheAccess.DEFAULT_QUEUE_SIZE;
import static org.neo4j.consistency.checking.full.CloningRecordIterator.cloned;
import static org.neo4j.consistency.checking.full.RecordDistributor.distributeRecords;
import static org.neo4j.kernel.impl.store.Scanner.scan;

/**
 * Full check works by spawning StoreProcessorTasks that call StoreProcessor. StoreProcessor.applyFiltered()
 * then scans the store and in turn calls down to store.accept which then knows how to check the given record.
 */
public class StoreProcessor extends AbstractStoreProcessor
{
    protected final CacheAccess cacheAccess;
    private final ConsistencyReport.Reporter report;
    private SchemaRecordCheck schemaRecordCheck;
    private final Stage stage;

    public StoreProcessor( CheckDecorator decorator, ConsistencyReport.Reporter report,
            Stage stage, CacheAccess cacheAccess )
    {
        super( decorator );
        assert stage != null;
        this.report = report;
        this.stage = stage;
        this.cacheAccess = cacheAccess;
    }

    public Stage getStage()
    {
        return stage;
    }

    @Override
    public void processNode( RecordStore<NodeRecord> store, NodeRecord node, PageCursorTracer cursorTracer )
    {
        cacheAccess.client().incAndGetCount( node.isDense() ? Counts.Type.nodeDense : Counts.Type.nodeSparse );
        super.processNode( store, node, cursorTracer );
    }

    @Override
    protected void checkNode( RecordStore<NodeRecord> store, NodeRecord node,
            RecordCheck<NodeRecord,ConsistencyReport.NodeConsistencyReport> checker, PageCursorTracer cursorTracer )
    {
        report.forNode( node, checker, cursorTracer );
    }

    private void countLinks( long id1, long id2, CacheAccess.Client client )
    {
        Counts.Type type;
        if ( id2 == -1 )
        {
            type = Counts.Type.nullLinks;
        }
        else if ( id2 > id1 )
        {
            type = Counts.Type.forwardLinks;
        }
        else
        {
            type = Counts.Type.backLinks;
        }
        client.incAndGetCount( type );
    }

    @Override
    protected void checkRelationship( RecordStore<RelationshipRecord> store, RelationshipRecord rel,
                                      RecordCheck<RelationshipRecord,ConsistencyReport.RelationshipConsistencyReport> checker,
            PageCursorTracer cursorTracer )
    {
        if ( stage != null && (stage == CheckStage.Stage6_RS_Forward || stage == CheckStage.Stage7_RS_Backward) )
        {
            long id = rel.getId();
            CacheAccess.Client client = cacheAccess.client();
            countLinks( id, rel.getFirstNextRel(), client );
            countLinks( id, rel.getFirstPrevRel(), client );
            countLinks( id, rel.getSecondNextRel(), client );
            countLinks( id, rel.getSecondPrevRel(), client );
        }
        report.forRelationship( rel, checker, cursorTracer );
    }

    @Override
    protected void checkProperty( RecordStore<PropertyRecord> store, PropertyRecord property,
            RecordCheck<PropertyRecord,ConsistencyReport.PropertyConsistencyReport> checker, PageCursorTracer cursorTracer )
    {
        report.forProperty( property, checker, cursorTracer );
    }

    @Override
    protected void checkRelationshipTypeToken( RecordStore<RelationshipTypeTokenRecord> store,
                                               RelationshipTypeTokenRecord relationshipType,
                                               RecordCheck<RelationshipTypeTokenRecord,
                                                       ConsistencyReport.RelationshipTypeConsistencyReport> checker, PageCursorTracer cursorTracer )
    {
        report.forRelationshipTypeName( relationshipType, checker, cursorTracer );
    }

    @Override
    protected void checkLabelToken( RecordStore<LabelTokenRecord> store, LabelTokenRecord label,
                                    RecordCheck<LabelTokenRecord, ConsistencyReport.LabelTokenConsistencyReport> checker, PageCursorTracer cursorTracer )
    {
        report.forLabelName( label, checker, cursorTracer );
    }

    @Override
    protected void checkPropertyKeyToken( RecordStore<PropertyKeyTokenRecord> store, PropertyKeyTokenRecord key,
                                          RecordCheck<PropertyKeyTokenRecord,
                                          ConsistencyReport.PropertyKeyTokenConsistencyReport> checker, PageCursorTracer cursorTracer )
    {
        report.forPropertyKey( key, checker, cursorTracer);
    }

    @Override
    protected void checkDynamic( RecordType type, RecordStore<DynamicRecord> store, DynamicRecord string,
                                 RecordCheck<DynamicRecord,ConsistencyReport.DynamicConsistencyReport> checker, PageCursorTracer cursorTracer )
    {
        report.forDynamicBlock( type, string, checker, cursorTracer );
    }

    @Override
    protected void checkDynamicLabel( RecordType type, RecordStore<DynamicRecord> store, DynamicRecord string,
                                      RecordCheck<DynamicRecord,DynamicLabelConsistencyReport> checker, PageCursorTracer cursorTracer )
    {
        report.forDynamicLabelBlock( type, string, checker, cursorTracer );
    }

    @Override
    protected void checkRelationshipGroup( RecordStore<RelationshipGroupRecord> store, RelationshipGroupRecord record,
            RecordCheck<RelationshipGroupRecord,RelationshipGroupConsistencyReport> checker, PageCursorTracer cursorTracer )
    {
        report.forRelationshipGroup( record, checker, cursorTracer );
    }

    void setSchemaRecordCheck( SchemaRecordCheck schemaRecordCheck )
    {
        this.schemaRecordCheck = schemaRecordCheck;
    }

    @Override
    public void processSchema( RecordStore<SchemaRecord> store, SchemaRecord schema, PageCursorTracer cursorTracer )
    {
        if ( null == schemaRecordCheck )
        {
            super.processSchema( store, schema, cursorTracer );
        }
        else
        {
            report.forSchema( schema, schemaRecordCheck, cursorTracer );
        }
    }

    <R extends AbstractBaseRecord> void applyFilteredParallel( RecordStore<R> store, ProgressListener progressListener, int numberOfThreads, long recordsPerCpu,
            final QueueDistributor<R> distributor, PageCacheTracer pageCacheTracer )
    {
        cacheAccess.prepareForProcessingOfSingleStore( recordsPerCpu );
        RecordProcessor<R> processor = new RecordProcessor.Adapter<>()
        {
            @Override
            public void init( int id )
            {
                // Thread id assignment happens here, so do this before processing. Calles to this init
                // method is ordered externally.
                cacheAccess.client();
            }

            @Override
            public void process( R record, PageCursorTracer cursorTracer )
            {
                store.accept( StoreProcessor.this, record, cursorTracer );
            }
        };

        ResourceIterable<R> scan = scan( store, stage.isForward(), pageCacheTracer );
        try ( ResourceIterator<R> records = scan.iterator() )
        {
            distributeRecords( numberOfThreads, getClass().getSimpleName(), DEFAULT_QUEUE_SIZE, cloned( records ), progressListener, processor, distributor,
                    pageCacheTracer );
        }
    }
}
