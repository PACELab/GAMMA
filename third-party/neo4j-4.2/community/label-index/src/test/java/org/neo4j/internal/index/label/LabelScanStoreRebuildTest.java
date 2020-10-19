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
package org.neo4j.internal.index.label;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.neo4j.io.fs.FileSystemAbstraction;
import org.neo4j.io.layout.DatabaseLayout;
import org.neo4j.io.pagecache.PageCache;
import org.neo4j.io.pagecache.tracing.PageCacheTracer;
import org.neo4j.monitoring.Monitors;
import org.neo4j.storageengine.api.EntityTokenUpdate;
import org.neo4j.test.extension.Inject;
import org.neo4j.test.extension.Neo4jLayoutExtension;
import org.neo4j.test.extension.pagecache.PageCacheExtension;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.neo4j.index.internal.gbptree.RecoveryCleanupWorkCollector.ignore;
import static org.neo4j.index.internal.gbptree.RecoveryCleanupWorkCollector.immediate;
import static org.neo4j.internal.index.label.FullStoreChangeStream.EMPTY;
import static org.neo4j.internal.index.label.FullStoreChangeStream.asStream;
import static org.neo4j.internal.index.label.TokenScanStore.labelScanStore;
import static org.neo4j.memory.EmptyMemoryTracker.INSTANCE;

@PageCacheExtension
@Neo4jLayoutExtension
class LabelScanStoreRebuildTest
{
    @Inject
    private DatabaseLayout databaseLayout;
    @Inject
    private FileSystemAbstraction fileSystem;
    @Inject
    private PageCache pageCache;

    private static final FullStoreChangeStream THROWING_STREAM = ( writer, cursorTracer, memoryTracker ) ->
    {
        throw new IllegalArgumentException();
    };

    @Test
    void mustBeDirtyIfFailedDuringRebuild() throws Exception
    {
        // given
        createDirtyIndex( pageCache );

        // when
        RecordingMonitor monitor = new RecordingMonitor();
        Monitors monitors = new Monitors();
        monitors.addMonitorListener( monitor );

        LabelScanStore labelScanStore = labelScanStore( pageCache, databaseLayout, fileSystem, EMPTY, false, monitors, immediate(),
                PageCacheTracer.NULL, INSTANCE );
        labelScanStore.init();
        labelScanStore.start();

        // then
        assertTrue( monitor.notValid );
        assertTrue( monitor.rebuilding );
        assertTrue( monitor.rebuilt );
        labelScanStore.shutdown();
    }

    @Test
    void doNotRebuildIfOpenedInReadOnlyModeAndIndexIsNotClean() throws IOException
    {
        createDirtyIndex( pageCache );

        Monitors monitors = new Monitors();
        RecordingMonitor monitor = new RecordingMonitor();
        monitors.addMonitorListener( monitor );

        LabelScanStore labelScanStore = labelScanStore( pageCache, databaseLayout, fileSystem, EMPTY, true, monitors, ignore(), PageCacheTracer.NULL,
                INSTANCE );
        labelScanStore.init();
        labelScanStore.start();

        assertTrue( monitor.notValid );
        assertFalse( monitor.rebuilt );
        assertFalse( monitor.rebuilding );
        labelScanStore.shutdown();
    }

    @Test
    void shouldFailOnUnsortedLabelsFromFullStoreChangeStream() throws Exception
    {
        // given
        List<EntityTokenUpdate> existingData = new ArrayList<>();
        existingData.add( EntityTokenUpdate.tokenChanges( 1, new long[0], new long[]{2, 1} ) );
        FullStoreChangeStream changeStream = asStream( existingData );
        LabelScanStore labelScanStore = labelScanStore( pageCache, databaseLayout, fileSystem, changeStream, false, new Monitors(), immediate(),
                PageCacheTracer.NULL, INSTANCE );
        try
        {
            labelScanStore.init();
            IllegalArgumentException exception = assertThrows( IllegalArgumentException.class, labelScanStore::start );
            assertThat( exception.getMessage() ).contains( "unsorted tokens" );
            assertThat( exception.getMessage() ).containsSubsequence( "2", "1" );
        }
        finally
        {
            labelScanStore.shutdown();
        }
    }

    private void createDirtyIndex( PageCache pageCache ) throws IOException
    {
        LabelScanStore labelScanStore = null;
        try
        {
            labelScanStore = labelScanStore( pageCache, databaseLayout, fileSystem, THROWING_STREAM, false, new Monitors(), immediate(),
                    PageCacheTracer.NULL, INSTANCE );

            labelScanStore.init();
            labelScanStore.start();
        }
        catch ( IllegalArgumentException e )
        {
            if ( labelScanStore != null )
            {
                labelScanStore.shutdown();
            }
        }
    }

    private static class RecordingMonitor extends LabelScanStore.Monitor.Adaptor
    {
        boolean notValid;
        boolean rebuilding;
        boolean rebuilt;

        @Override
        public void notValidIndex()
        {
            notValid = true;
        }

        @Override
        public void rebuilding()
        {
            rebuilding = true;
        }

        @Override
        public void rebuilt( long roughEntityCount )
        {
            rebuilt = true;
        }
    }
}
