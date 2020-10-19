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

import java.util.Iterator;

import org.neo4j.graphdb.ResourceIterable;
import org.neo4j.graphdb.ResourceIterator;
import org.neo4j.internal.helpers.collection.BoundedIterable;
import org.neo4j.io.pagecache.tracing.PageCacheTracer;
import org.neo4j.kernel.impl.store.RecordStore;
import org.neo4j.kernel.impl.store.record.AbstractBaseRecord;

import static org.neo4j.consistency.checking.full.CloningRecordIterator.cloned;
import static org.neo4j.kernel.impl.store.Scanner.scan;
import static org.neo4j.kernel.impl.store.record.RecordLoad.FORCE;

public class IterableStore<RECORD extends AbstractBaseRecord> implements BoundedIterable<RECORD>
{
    private static final String CONSISTENCY_CHECKER_CACHE_WARM_UP_TAG = "consistencyCheckerCacheWarmUp";
    private final RecordStore<RECORD> store;
    private final boolean forward;
    private final PageCacheTracer pageCacheTracer;
    private ResourceIterator<RECORD> iterator;

    public IterableStore( RecordStore<RECORD> store, boolean forward, PageCacheTracer pageCacheTracer )
    {
        this.store = store;
        this.forward = forward;
        this.pageCacheTracer = pageCacheTracer;
    }

    @Override
    public long maxCount()
    {
        return store.getHighId();
    }

    @Override
    public void close()
    {
        closeIterator();
    }

    private void closeIterator()
    {
        if ( iterator != null )
        {
            iterator.close();
            iterator = null;
        }
    }

    @Override
    public Iterator<RECORD> iterator()
    {
        closeIterator();
        ResourceIterable<RECORD> iterable = scan( store, forward, pageCacheTracer );
        return cloned( iterator = iterable.iterator() );
    }

    public void warmUpCache()
    {
        int recordsPerPage = store.getRecordsPerPage();
        long id = 0;
        long half = store.getHighId() / 2;
        RECORD record = store.newRecord();
        try ( var cursorTracer = pageCacheTracer.createPageCursorTracer( CONSISTENCY_CHECKER_CACHE_WARM_UP_TAG ) )
        {
            while ( id < half )
            {
                store.getRecord( id, record, FORCE, cursorTracer );
                id += recordsPerPage - 1;
            }
        }
    }
}
