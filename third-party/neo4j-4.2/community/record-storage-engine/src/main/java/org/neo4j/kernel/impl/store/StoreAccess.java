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
package org.neo4j.kernel.impl.store;

import org.neo4j.configuration.Config;
import org.neo4j.internal.id.DefaultIdGeneratorFactory;
import org.neo4j.io.fs.FileSystemAbstraction;
import org.neo4j.io.layout.DatabaseLayout;
import org.neo4j.io.pagecache.PageCache;
import org.neo4j.io.pagecache.tracing.PageCacheTracer;
import org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer;
import org.neo4j.kernel.impl.store.record.AbstractBaseRecord;
import org.neo4j.kernel.impl.store.record.DynamicRecord;
import org.neo4j.kernel.impl.store.record.LabelTokenRecord;
import org.neo4j.kernel.impl.store.record.NodeRecord;
import org.neo4j.kernel.impl.store.record.PropertyKeyTokenRecord;
import org.neo4j.kernel.impl.store.record.PropertyRecord;
import org.neo4j.kernel.impl.store.record.RelationshipGroupRecord;
import org.neo4j.kernel.impl.store.record.RelationshipRecord;
import org.neo4j.kernel.impl.store.record.RelationshipTypeTokenRecord;
import org.neo4j.logging.NullLogProvider;

import static org.neo4j.index.internal.gbptree.RecoveryCleanupWorkCollector.immediate;

/**
 * Not thread safe (since DiffRecordStore is not thread safe), intended for
 * single threaded use.
 *
 * Make sure to call {@link #initialize()} after constructor has been run.
 */
public class StoreAccess
{
    // Top level stores
    private SchemaStore schemaStore;
    private RecordStore<NodeRecord> nodeStore;
    private RecordStore<RelationshipRecord> relStore;
    private RecordStore<RelationshipTypeTokenRecord> relationshipTypeTokenStore;
    private RecordStore<LabelTokenRecord> labelTokenStore;
    private RecordStore<DynamicRecord> nodeDynamicLabelStore;
    private RecordStore<PropertyRecord> propStore;
    // Transitive stores
    private RecordStore<DynamicRecord> stringStore;
    private RecordStore<DynamicRecord> arrayStore;
    private RecordStore<PropertyKeyTokenRecord> propertyKeyTokenStore;
    private RecordStore<DynamicRecord> relationshipTypeNameStore;
    private RecordStore<DynamicRecord> labelNameStore;
    private RecordStore<DynamicRecord> propertyKeyNameStore;
    private RecordStore<RelationshipGroupRecord> relGroupStore;
    // internal state
    private boolean closeable;
    private final NeoStores neoStores;

    public StoreAccess( NeoStores store )
    {
        this.neoStores = store;
    }

    public StoreAccess( FileSystemAbstraction fileSystem, PageCache pageCache, DatabaseLayout directoryStructure, Config config, PageCacheTracer cacheTracer )
    {
        this( new StoreFactory( directoryStructure, config, new DefaultIdGeneratorFactory( fileSystem, immediate() ), pageCache,
                fileSystem, NullLogProvider.getInstance(), cacheTracer ).openAllNeoStores() );
        this.closeable = true;
    }

    /**
     * This method exists since {@link #wrapStore(RecordStore)} might depend on the existence of a variable
     * that gets set in a subclass' constructor <strong>after</strong> this constructor of {@link StoreAccess}
     * has been executed. I.e. a correct creation of a {@link StoreAccess} instance must be the creation of the
     * object plus a call to this method.
     *
     * @return this
     */
    public StoreAccess initialize()
    {
        // Wrap stores to count access
        this.schemaStore = neoStores.getSchemaStore();
        this.nodeStore = wrapStore( neoStores.getNodeStore() );
        this.relStore = wrapStore( neoStores.getRelationshipStore() );
        this.propStore = wrapStore( neoStores.getPropertyStore() );
        this.stringStore = wrapStore( neoStores.getPropertyStore().getStringStore() );
        this.arrayStore = wrapStore( neoStores.getPropertyStore().getArrayStore() );
        this.relationshipTypeTokenStore = wrapStore( neoStores.getRelationshipTypeTokenStore() );
        this.labelTokenStore = wrapStore( neoStores.getLabelTokenStore() );
        this.nodeDynamicLabelStore = wrapStore( wrapNodeDynamicLabelStore( neoStores.getNodeStore().getDynamicLabelStore() ) );
        this.propertyKeyTokenStore = wrapStore( neoStores.getPropertyStore().getPropertyKeyTokenStore() );
        this.relationshipTypeNameStore = wrapStore( neoStores.getRelationshipTypeTokenStore().getNameStore() );
        this.labelNameStore = wrapStore( neoStores.getLabelTokenStore().getNameStore() );
        this.propertyKeyNameStore = wrapStore( neoStores.getPropertyStore().getPropertyKeyTokenStore().getNameStore() );
        this.relGroupStore = wrapStore( neoStores.getRelationshipGroupStore() );

        return this;
    }

    public NeoStores getRawNeoStores()
    {
        return neoStores;
    }

    public SchemaStore getSchemaStore()
    {
        return schemaStore;
    }

    public RecordStore<NodeRecord> getNodeStore()
    {
        return nodeStore;
    }

    public RecordStore<RelationshipRecord> getRelationshipStore()
    {
        return relStore;
    }

    public RecordStore<RelationshipGroupRecord> getRelationshipGroupStore()
    {
        return relGroupStore;
    }

    public RecordStore<PropertyRecord> getPropertyStore()
    {
        return propStore;
    }

    public RecordStore<DynamicRecord> getStringStore()
    {
        return stringStore;
    }

    public RecordStore<DynamicRecord> getArrayStore()
    {
        return arrayStore;
    }

    public RecordStore<RelationshipTypeTokenRecord> getRelationshipTypeTokenStore()
    {
        return relationshipTypeTokenStore;
    }

    public RecordStore<LabelTokenRecord> getLabelTokenStore()
    {
        return labelTokenStore;
    }

    public RecordStore<DynamicRecord> getNodeDynamicLabelStore()
    {
        return nodeDynamicLabelStore;
    }

    public RecordStore<PropertyKeyTokenRecord> getPropertyKeyTokenStore()
    {
        return propertyKeyTokenStore;
    }

    public RecordStore<DynamicRecord> getRelationshipTypeNameStore()
    {
        return relationshipTypeNameStore;
    }

    public RecordStore<DynamicRecord> getLabelNameStore()
    {
        return labelNameStore;
    }

    public RecordStore<DynamicRecord> getPropertyKeyNameStore()
    {
        return propertyKeyNameStore;
    }

    private static RecordStore<DynamicRecord> wrapNodeDynamicLabelStore( RecordStore<DynamicRecord> store )
    {
        return new RecordStore.Delegator<>( store )
        {
            @Override
            public <FAILURE extends Exception> void accept( Processor<FAILURE> processor, DynamicRecord record, PageCursorTracer cursorTracer )
                    throws FAILURE
            {
                processor.processLabelArrayWithOwner( this, record, cursorTracer );
            }
        };
    }

    protected <R extends AbstractBaseRecord> RecordStore<R> wrapStore( RecordStore<R> store )
    {
        return store;
    }

    public synchronized void close()
    {
        if ( closeable )
        {
            closeable = false;
            neoStores.close();
        }
    }
}
