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

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.ArrayList;
import java.util.List;

import org.neo4j.configuration.Config;
import org.neo4j.gis.spatial.index.curves.SpaceFillingCurve;
import org.neo4j.gis.spatial.index.curves.StandardConfiguration;
import org.neo4j.index.internal.gbptree.RecoveryCleanupWorkCollector;
import org.neo4j.internal.kernel.api.IndexQuery;
import org.neo4j.internal.kernel.api.QueryContext;
import org.neo4j.internal.kernel.api.exceptions.schema.IndexNotApplicableKernelException;
import org.neo4j.internal.schema.IndexDescriptor;
import org.neo4j.io.fs.FileSystemAbstraction;
import org.neo4j.io.pagecache.PageCache;
import org.neo4j.kernel.api.exceptions.index.IndexEntryConflictException;
import org.neo4j.kernel.api.index.IndexDirectoryStructure;
import org.neo4j.kernel.api.index.IndexReader;
import org.neo4j.kernel.api.schema.index.TestIndexDescriptorFactory;
import org.neo4j.kernel.impl.api.index.IndexUpdateMode;
import org.neo4j.kernel.impl.index.schema.config.IndexSpecificSpaceFillingCurveSettings;
import org.neo4j.storageengine.api.IndexEntryUpdate;
import org.neo4j.storageengine.api.schema.SimpleNodeValueClient;
import org.neo4j.test.extension.Inject;
import org.neo4j.test.extension.RandomExtension;
import org.neo4j.test.extension.pagecache.PageCacheExtension;
import org.neo4j.test.rule.RandomRule;
import org.neo4j.test.rule.TestDirectory;
import org.neo4j.values.storable.CoordinateReferenceSystem;
import org.neo4j.values.storable.PointArray;
import org.neo4j.values.storable.PointValue;
import org.neo4j.values.storable.Value;
import org.neo4j.values.storable.Values;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.neo4j.internal.kernel.api.IndexQueryConstraints.unorderedValues;
import static org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer.NULL;
import static org.neo4j.kernel.impl.index.schema.GenericNativeIndexProvider.DESCRIPTOR;
import static org.neo4j.values.storable.CoordinateReferenceSystem.WGS84;

@PageCacheExtension
@ExtendWith( RandomExtension.class )
class GenericAccessorPointsTest
{
    private static final CoordinateReferenceSystem crs = CoordinateReferenceSystem.WGS84;
    private static final Config config = Config.defaults();
    private static final IndexSpecificSpaceFillingCurveSettings indexSettings = IndexSpecificSpaceFillingCurveSettings.fromConfig( config );
    private static final SpaceFillingCurve curve = indexSettings.forCrs( crs );

    @Inject
    private FileSystemAbstraction fs;
    @Inject
    private TestDirectory directory;
    @Inject
    private PageCache pageCache;
    @Inject
    private RandomRule random;

    private NativeIndexAccessor accessor;
    private IndexDescriptor descriptor;

    @BeforeEach
    void setup()
    {
        IndexDirectoryStructure directoryStructure = IndexDirectoryStructure.directoriesByProvider( directory.homePath() ).forProvider( DESCRIPTOR );
        descriptor = TestIndexDescriptorFactory.forLabel( 1, 1 );
        IndexFiles indexFiles = new IndexFiles( fs, directoryStructure, descriptor.getId() );
        GenericLayout layout = new GenericLayout( 1, indexSettings );
        RecoveryCleanupWorkCollector collector = RecoveryCleanupWorkCollector.ignore();
        DatabaseIndexContext databaseIndexContext = DatabaseIndexContext.builder( pageCache, fs ).build();
        StandardConfiguration configuration = new StandardConfiguration();
        accessor = new GenericNativeIndexAccessor( databaseIndexContext, indexFiles, layout, collector, descriptor, indexSettings, configuration );
    }

    @AfterEach
    void tearDown()
    {
        accessor.close();
    }

    /**
     * This test verify that we correctly handle unique points that all belong to the same tile on the space filling curve.
     * All points share at least one dimension coordinate with another point to exercise minimal splitter.
     * We verify this by asserting that we always get exactly one hit on an exact match and that the value is what we expect.
     */
    @Test
    void mustHandlePointsWithinSameTile() throws IndexEntryConflictException, IndexNotApplicableKernelException
    {
        // given
        // Many random points that all are close enough to each other to belong to the same tile on the space filling curve.
        int nbrOfValues = 10000;
        PointValue origin = Values.pointValue( WGS84, 0.0, 0.0 );
        Long derivedValueForCenterPoint = curve.derivedValueFor( origin.coordinate() );
        double[] centerPoint = curve.centerPointFor( derivedValueForCenterPoint );
        double xWidthMultiplier = curve.getTileWidth( 0, curve.getMaxLevel() ) / 2;
        double yWidthMultiplier = curve.getTileWidth( 1, curve.getMaxLevel() ) / 2;

        List<Value> pointValues = new ArrayList<>();
        List<IndexEntryUpdate<?>> updates = new ArrayList<>();
        long nodeId = 1;
        for ( int i = 0; i < nbrOfValues / 4; i++ )
        {
            double x1 = (random.nextDouble() * 2 - 1) * xWidthMultiplier;
            double x2 = (random.nextDouble() * 2 - 1) * xWidthMultiplier;
            double y1 = (random.nextDouble() * 2 - 1) * yWidthMultiplier;
            double y2 = (random.nextDouble() * 2 - 1) * yWidthMultiplier;
            PointValue value11 = Values.pointValue( WGS84, centerPoint[0] + x1, centerPoint[1] + y1 );
            PointValue value12 = Values.pointValue( WGS84, centerPoint[0] + x1, centerPoint[1] + y2 );
            PointValue value21 = Values.pointValue( WGS84, centerPoint[0] + x2, centerPoint[1] + y1 );
            PointValue value22 = Values.pointValue( WGS84, centerPoint[0] + x2, centerPoint[1] + y2 );
            assertDerivedValue( derivedValueForCenterPoint, value11, value12, value21, value22 );

            nodeId = addPointsToLists( pointValues, updates, nodeId, value11, value12, value21, value22 );
        }

        processAll( updates );

        // then
        exactMatchOnAllValues( pointValues );
    }

    /**
     * This test verify that we correctly handle unique point arrays where every point in every array belong to the same tile on the space filling curve.
     * We verify this by asserting that we always get exactly one hit on an exact match and that the value is what we expect.
     */
    @Test
    void mustHandlePointArraysWithinSameTile() throws IndexEntryConflictException, IndexNotApplicableKernelException
    {
        // given
        // Many random points that all are close enough to each other to belong to the same tile on the space filling curve.
        int nbrOfValues = 10000;
        PointValue origin = Values.pointValue( WGS84, 0.0, 0.0 );
        Long derivedValueForCenterPoint = curve.derivedValueFor( origin.coordinate() );
        double[] centerPoint = curve.centerPointFor( derivedValueForCenterPoint );
        double xWidthMultiplier = curve.getTileWidth( 0, curve.getMaxLevel() ) / 2;
        double yWidthMultiplier = curve.getTileWidth( 1, curve.getMaxLevel() ) / 2;

        List<Value> pointArrays = new ArrayList<>();
        List<IndexEntryUpdate<?>> updates = new ArrayList<>();
        for ( int i = 0; i < nbrOfValues; i++ )
        {
            int arrayLength = random.nextInt( 5 ) + 1;
            PointValue[] pointValues = new PointValue[arrayLength];
            for ( int j = 0; j < arrayLength; j++ )
            {
                double x = (random.nextDouble() * 2 - 1) * xWidthMultiplier;
                double y = (random.nextDouble() * 2 - 1) * yWidthMultiplier;
                PointValue value = Values.pointValue( WGS84, centerPoint[0] + x, centerPoint[1] + y );

                assertDerivedValue( derivedValueForCenterPoint, value );
                pointValues[j] = value;
            }
            PointArray array = Values.pointArray( pointValues );
            pointArrays.add( array );
            updates.add( IndexEntryUpdate.add( i, descriptor, array ) );
        }

        processAll( updates );

        // then
        exactMatchOnAllValues( pointArrays );
    }

    /**
     * The test mustHandlePointArraysWithinSameTile was flaky on random numbers that placed points just
     * within the tile upper bound, and allocated points to adjacent tiles due to rounding errors.
     * This test uses a specific point that triggers that exact failure in a non-flaky way.
     */
    @Test
    void shouldNotGetRoundingErrorsWithPointsJustWithinTheTileUpperBound()
    {
        PointValue origin = Values.pointValue( WGS84, 0.0, 0.0 );
        long derivedValueForCenterPoint = curve.derivedValueFor( origin.coordinate() );
        double[] centerPoint = curve.centerPointFor( derivedValueForCenterPoint ); // [1.6763806343078613E-7, 8.381903171539307E-8]

        double xWidthMultiplier = curve.getTileWidth( 0, curve.getMaxLevel() ) / 2; // 1.6763806343078613E-7
        double yWidthMultiplier = curve.getTileWidth( 1, curve.getMaxLevel() ) / 2; // 8.381903171539307E-8

        double[] faultyCoords = {1.874410632171803E-8, 1.6763806281859016E-7};

        assertTrue( centerPoint[0] + xWidthMultiplier > faultyCoords[0], "inside upper x limit" );
        assertTrue( centerPoint[0] - xWidthMultiplier < faultyCoords[0], "inside lower x limit" );

        assertTrue( centerPoint[1] + yWidthMultiplier > faultyCoords[1], "inside upper y limit" );
        assertTrue( centerPoint[1] - yWidthMultiplier < faultyCoords[1], "inside lower y limit" );

        long derivedValueForFaultyCoords = curve.derivedValueFor( faultyCoords );
        assertEquals( derivedValueForCenterPoint, derivedValueForFaultyCoords, "expected same derived value" );
    }

    private long addPointsToLists( List<Value> pointValues, List<IndexEntryUpdate<?>> updates, long nodeId, PointValue... values )
    {
        for ( PointValue value : values )
        {
            pointValues.add( value );
            updates.add( IndexEntryUpdate.add( nodeId++, descriptor, value ) );
        }
        return nodeId;
    }

    private static void assertDerivedValue( Long targetDerivedValue, PointValue... values )
    {
        for ( PointValue value : values )
        {
            Long derivedValueForValue = curve.derivedValueFor( value.coordinate() );
            assertEquals( targetDerivedValue, derivedValueForValue, "expected random value to belong to same tile as center point" );
        }
    }

    private void processAll( List<IndexEntryUpdate<?>> updates ) throws IndexEntryConflictException
    {
        try ( NativeIndexUpdater updater = accessor.newUpdater( IndexUpdateMode.ONLINE, NULL ) )
        {
            for ( IndexEntryUpdate<?> update : updates )
            {
                //noinspection unchecked
                updater.process( update );
            }
        }
    }

    private void exactMatchOnAllValues( List<Value> values ) throws IndexNotApplicableKernelException
    {
        try ( IndexReader indexReader = accessor.newReader() )
        {
            SimpleNodeValueClient client = new SimpleNodeValueClient();
            for ( Value value : values )
            {
                IndexQuery.ExactPredicate exact = IndexQuery.exact( descriptor.schema().getPropertyId(), value );
                indexReader.query( QueryContext.NULL_CONTEXT, client, unorderedValues(), exact );

                // then
                assertTrue( client.next() );
                assertEquals( value, client.values[0] );
                assertFalse( client.next() );
            }
        }
    }
}
