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
package org.neo4j.consistency.checking.index;

import org.neo4j.common.TokenNameLookup;
import org.neo4j.consistency.checking.full.IndexCheck;
import org.neo4j.consistency.checking.full.RecordProcessor;
import org.neo4j.consistency.report.ConsistencyReporter;
import org.neo4j.consistency.store.synthetic.IndexEntry;
import org.neo4j.internal.schema.IndexDescriptor;
import org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer;

public class IndexEntryProcessor extends RecordProcessor.Adapter<Long>
{
    private final ConsistencyReporter reporter;
    private final IndexCheck indexCheck;
    private final IndexDescriptor indexDescriptor;
    private final TokenNameLookup tokenNameLookup;

    public IndexEntryProcessor( ConsistencyReporter reporter, IndexCheck indexCheck, IndexDescriptor indexDescriptor, TokenNameLookup tokenNameLookup )
    {
        this.reporter = reporter;
        this.indexCheck = indexCheck;
        this.indexDescriptor = indexDescriptor;
        this.tokenNameLookup = tokenNameLookup;
    }

    @Override
    public void process( Long nodeId, PageCursorTracer cursorTracer )
    {
        reporter.forIndexEntry( new IndexEntry( indexDescriptor, tokenNameLookup, nodeId ), indexCheck, cursorTracer );
    }
}
