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
package org.neo4j.kernel.impl.index.schema;

import org.apache.commons.lang3.StringUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;

import org.neo4j.configuration.Config;
import org.neo4j.index.internal.gbptree.RecoveryCleanupWorkCollector;
import org.neo4j.internal.kernel.api.InternalIndexState;
import org.neo4j.internal.schema.IndexDescriptor;
import org.neo4j.io.fs.FileSystemAbstraction;
import org.neo4j.io.pagecache.PageCache;
import org.neo4j.kernel.api.exceptions.index.IndexEntryConflictException;
import org.neo4j.kernel.api.index.IndexAccessor;
import org.neo4j.kernel.api.index.IndexDirectoryStructure;
import org.neo4j.kernel.api.index.IndexPopulator;
import org.neo4j.kernel.api.index.IndexProvider;
import org.neo4j.kernel.api.index.IndexUpdater;
import org.neo4j.kernel.api.index.LoggingMonitor;
import org.neo4j.kernel.impl.api.index.IndexSamplingConfig;
import org.neo4j.kernel.impl.api.index.IndexUpdateMode;
import org.neo4j.logging.AssertableLogProvider;
import org.neo4j.storageengine.api.IndexEntryUpdate;
import org.neo4j.test.extension.Inject;
import org.neo4j.test.extension.pagecache.EphemeralPageCacheExtension;
import org.neo4j.test.rule.TestDirectory;
import org.neo4j.values.storable.Value;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.neo4j.index.internal.gbptree.RecoveryCleanupWorkCollector.immediate;
import static org.neo4j.internal.schema.IndexPrototype.forSchema;
import static org.neo4j.internal.schema.IndexPrototype.uniqueForSchema;
import static org.neo4j.internal.schema.SchemaDescriptor.forLabel;
import static org.neo4j.io.memory.ByteBufferFactory.heapBufferFactory;
import static org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer.NULL;
import static org.neo4j.kernel.api.index.IndexDirectoryStructure.directoriesByProvider;
import static org.neo4j.kernel.impl.api.index.TestIndexProviderDescriptor.PROVIDER_DESCRIPTOR;
import static org.neo4j.logging.LogAssertions.assertThat;
import static org.neo4j.memory.EmptyMemoryTracker.INSTANCE;

@EphemeralPageCacheExtension
abstract class NativeIndexProviderTests
{
    private static final int indexId = 1;
    private static final int labelId = 1;
    private static final int propId = 1;

    @Inject
    private PageCache pageCache;
    @Inject
    private TestDirectory testDirectory;
    @Inject
    private FileSystemAbstraction fs;

    private final AssertableLogProvider logging = new AssertableLogProvider();
    private final IndexProvider.Monitor monitor = new LoggingMonitor( logging.getLog( "test" ) );
    private final ProviderFactory factory;
    private final InternalIndexState expectedStateOnNonExistingSubIndex;
    private final Value someValue;
    private IndexProvider provider;

    NativeIndexProviderTests( ProviderFactory factory, InternalIndexState expectedStateOnNonExistingSubIndex, Value someValue )
    {
        this.factory = factory;
        this.expectedStateOnNonExistingSubIndex = expectedStateOnNonExistingSubIndex;
        this.someValue = someValue;
    }

    @BeforeEach
    void setup() throws IOException
    {
        Path nativeSchemaIndexStoreDirectory = newProvider().directoryStructure().rootDirectory();
        fs.mkdirs( nativeSchemaIndexStoreDirectory.toFile() );
    }

    /* getPopulator */

    @Test
    void getPopulatorMustThrowIfInReadOnlyMode()
    {
        // given
        provider = newReadOnlyProvider();

        assertThrows( UnsupportedOperationException.class, () -> provider.getPopulator( descriptor(), samplingConfig(),
                heapBufferFactory( 1024 ), INSTANCE ) );
    }

    /* getOnlineAccessor */

    @Test
    void shouldNotCheckConflictsWhenApplyingUpdatesInOnlineAccessor() throws IOException, IndexEntryConflictException
    {
        // given
        provider = newProvider();

        // when
        IndexDescriptor descriptor = descriptorUnique();
        try ( IndexAccessor accessor = provider.getOnlineAccessor( descriptor, samplingConfig() );
            IndexUpdater indexUpdater = accessor.newUpdater( IndexUpdateMode.ONLINE, NULL ) )
        {
            indexUpdater.process( IndexEntryUpdate.add( 1, descriptor.schema(), someValue ) );

            // then
            // ... expect no failure on duplicate value
            indexUpdater.process( IndexEntryUpdate.add( 2, descriptor.schema(), someValue ) );
        }
    }

    /* getPopulationFailure */

    @Test
    void getPopulationFailureReturnEmptyStringIfNoFailure()
    {
        // given
        provider = newProvider();
        IndexPopulator populator = provider.getPopulator( descriptor(), samplingConfig(), heapBufferFactory( 1024 ), INSTANCE );
        populator.create();
        populator.close( true, NULL );

        // when
        // ... no failure on populator

        assertEquals( StringUtils.EMPTY, provider.getPopulationFailure( descriptor(), NULL ) );
    }

    @Test
    void getPopulationFailureReturnEmptyStringIfFailureOnOtherIndex()
    {
        // given
        provider = newProvider();

        int nonFailedIndexId = NativeIndexProviderTests.indexId;
        IndexPopulator nonFailedPopulator = provider.getPopulator( descriptor( nonFailedIndexId ), samplingConfig(),
                heapBufferFactory( 1024 ), INSTANCE );
        nonFailedPopulator.create();
        nonFailedPopulator.close( true, NULL );

        int failedIndexId = 2;
        IndexPopulator failedPopulator = provider.getPopulator( descriptor( failedIndexId ), samplingConfig(),
                heapBufferFactory( 1024 ), INSTANCE );
        failedPopulator.create();

        // when
        failedPopulator.markAsFailed( "failure" );
        failedPopulator.close( false, NULL );

        var populationFailure = provider.getPopulationFailure( descriptor( nonFailedIndexId ), NULL );
        assertEquals( StringUtils.EMPTY, populationFailure );
    }

    @Test
    void getPopulationFailureMustReturnReportedFailure()
    {
        // given
        provider = newProvider();
        IndexPopulator populator = provider.getPopulator( descriptor(), samplingConfig(),
                heapBufferFactory( 1024 ), INSTANCE );
        populator.create();

        // when
        String failureMessage = "fail";
        populator.markAsFailed( failureMessage );
        populator.close( false, NULL );

        // then
        String populationFailure = provider.getPopulationFailure( descriptor(), NULL );
        assertThat( populationFailure ).isEqualTo( failureMessage );
    }

    @Test
    void getPopulationFailureMustReturnReportedFailuresForDifferentIndexIds()
    {
        // given
        provider = newProvider();
        int first = 1;
        int second = 2;
        int third = 3;
        IndexPopulator firstPopulator = provider.getPopulator( descriptor( first ), samplingConfig(),
                heapBufferFactory( 1024 ), INSTANCE );
        firstPopulator.create();
        IndexPopulator secondPopulator = provider.getPopulator( descriptor( second ), samplingConfig(),
                heapBufferFactory( 1024 ), INSTANCE );
        secondPopulator.create();
        IndexPopulator thirdPopulator = provider.getPopulator( descriptor( third ), samplingConfig(),
                heapBufferFactory( 1024 ), INSTANCE );
        thirdPopulator.create();

        // when
        String firstFailure = "first failure";
        firstPopulator.markAsFailed( firstFailure );
        firstPopulator.close( false, NULL );
        secondPopulator.close( true, NULL );
        String thirdFailure = "third failure";
        thirdPopulator.markAsFailed( thirdFailure );
        thirdPopulator.close( false, NULL );

        // then
        assertThat( provider.getPopulationFailure( descriptor( first ), NULL ) ).isEqualTo( firstFailure );
        assertThat( provider.getPopulationFailure( descriptor( third ), NULL ) ).isEqualTo( thirdFailure );
        assertEquals( StringUtils.EMPTY, provider.getPopulationFailure( descriptor( second ), NULL ) );
    }

    @Test
    void getPopulationFailureMustPersistReportedFailure()
    {
        // given
        provider = newProvider();
        IndexPopulator populator = provider.getPopulator( descriptor(), samplingConfig(),
                heapBufferFactory( 1024 ), INSTANCE );
        populator.create();

        // when
        String failureMessage = "fail";
        populator.markAsFailed( failureMessage );
        populator.close( false, NULL );

        // then
        provider = newProvider();
        String populationFailure = provider.getPopulationFailure( descriptor(), NULL );
        assertThat( populationFailure ).isEqualTo( failureMessage );
    }

    /* getInitialState */
    // pattern: open populator, markAsFailed, close populator, getInitialState, getPopulationFailure

    @Test
    void shouldReportCorrectInitialStateIfIndexDoesntExist()
    {
        // given
        provider = newProvider();

        // when
        InternalIndexState state = provider.getInitialState( descriptor(), NULL );

        // then
        assertEquals( expectedStateOnNonExistingSubIndex, state );
        var logAssert = assertThat( logging );
        if ( InternalIndexState.POPULATING == expectedStateOnNonExistingSubIndex )
        {
            logAssert.containsMessages( "Failed to open index" );
        }
        else
        {
            logAssert.doesNotContainMessage( "Failed to open index" );
        }
    }

    @Test
    void shouldReportInitialStateAsPopulatingIfPopulationStartedButIncomplete()
    {
        // given
        provider = newProvider();
        IndexPopulator populator = provider.getPopulator( descriptor(), samplingConfig(),
                heapBufferFactory( 1024 ), INSTANCE );
        populator.create();

        // when
        InternalIndexState state = provider.getInitialState( descriptor(), NULL );

        // then
        assertEquals( InternalIndexState.POPULATING, state );
        populator.close( true, NULL );
    }

    @Test
    void shouldReportInitialStateAsFailedIfMarkedAsFailed()
    {
        // given
        provider = newProvider();
        IndexPopulator populator = provider.getPopulator( descriptor(), samplingConfig(), heapBufferFactory( 1024 ), INSTANCE );
        populator.create();
        populator.markAsFailed( "Just some failure" );
        populator.close( false, NULL );

        // when
        InternalIndexState state = provider.getInitialState( descriptor(), NULL );

        // then
        assertEquals( InternalIndexState.FAILED, state );
    }

    @Test
    void shouldReportInitialStateAsOnlineIfPopulationCompletedSuccessfully()
    {
        // given
        provider = newProvider();
        IndexPopulator populator = provider.getPopulator( descriptor(), samplingConfig(), heapBufferFactory( 1024 ), INSTANCE );
        populator.create();
        populator.close( true, NULL );

        // when
        InternalIndexState state = provider.getInitialState( descriptor(), NULL );

        // then
        assertEquals( InternalIndexState.ONLINE, state );
    }

    private IndexProvider newProvider( boolean readOnly )
    {
        return factory.create( pageCache, fs, directoriesByProvider( testDirectory.homePath().toAbsolutePath() ), monitor, immediate(), readOnly );
    }

    private IndexProvider newProvider()
    {
        return newProvider( false );
    }

    private IndexProvider newReadOnlyProvider()
    {
        return newProvider( true );
    }

    private static IndexSamplingConfig samplingConfig()
    {
        return new IndexSamplingConfig( Config.defaults() );
    }

    private IndexDescriptor descriptor()
    {
        return completeConfiguration( forSchema( forLabel( labelId, propId ), PROVIDER_DESCRIPTOR ).withName( "index" ).materialise( indexId ) );
    }

    private IndexDescriptor descriptor( long indexId )
    {
        return completeConfiguration( forSchema( forLabel( labelId, propId ), PROVIDER_DESCRIPTOR ).withName( "index_" + indexId ).materialise( indexId ) );
    }

    private IndexDescriptor descriptorUnique()
    {
        return completeConfiguration( uniqueForSchema( forLabel( labelId, propId ), PROVIDER_DESCRIPTOR ).withName( "constraint" ).materialise( indexId ) );
    }

    private IndexDescriptor completeConfiguration( IndexDescriptor indexDescriptor )
    {
        return provider.completeConfiguration( indexDescriptor );
    }

    @FunctionalInterface
    interface ProviderFactory
    {
        IndexProvider create( PageCache pageCache, FileSystemAbstraction fs, IndexDirectoryStructure.Factory dir, IndexProvider.Monitor monitor,
            RecoveryCleanupWorkCollector collector, boolean readOnly );
    }
}
