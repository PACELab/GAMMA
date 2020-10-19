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
package org.neo4j.kernel.api.impl.schema.sampler;

import org.apache.lucene.search.IndexSearcher;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.concurrent.TimeUnit;

import org.neo4j.internal.kernel.api.exceptions.schema.IndexNotFoundKernelException;
import org.neo4j.kernel.api.impl.schema.TaskCoordinator;
import org.neo4j.kernel.api.index.IndexSample;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer.NULL;

class UniqueDatabaseIndexSamplerTest
{
    private final IndexSearcher indexSearcher = mock( IndexSearcher.class, Mockito.RETURNS_DEEP_STUBS );
    private final TaskCoordinator taskControl = new TaskCoordinator( 0, TimeUnit.MILLISECONDS );

    @Test
    void uniqueSamplingUseDocumentsNumber() throws IndexNotFoundKernelException
    {
        when( indexSearcher.getIndexReader().numDocs() ).thenReturn( 17 );

        UniqueLuceneIndexSampler sampler = new UniqueLuceneIndexSampler( indexSearcher, taskControl.newInstance() );
        IndexSample sample = sampler.sampleIndex( NULL );
        assertEquals( 17, sample.indexSize() );
    }

    @Test
    void uniqueSamplingCancel()
    {
        when( indexSearcher.getIndexReader().numDocs() ).thenAnswer( invocation ->
        {
            taskControl.cancel();
            return 17;
        } );

        UniqueLuceneIndexSampler sampler = new UniqueLuceneIndexSampler( indexSearcher, taskControl.newInstance() );
        IndexNotFoundKernelException notFoundKernelException = assertThrows( IndexNotFoundKernelException.class, () -> sampler.sampleIndex( NULL ) );
        assertEquals( "Index dropped while sampling.", notFoundKernelException.getMessage() );
    }

}
