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
package org.neo4j.internal.recordstorage;

import org.eclipse.collections.api.factory.Sets;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.neo4j.configuration.Config;
import org.neo4j.internal.helpers.collection.IteratorWrapper;
import org.neo4j.internal.id.DefaultIdGeneratorFactory;
import org.neo4j.io.fs.FileSystemAbstraction;
import org.neo4j.io.layout.DatabaseLayout;
import org.neo4j.io.pagecache.PageCache;
import org.neo4j.io.pagecache.tracing.PageCacheTracer;
import org.neo4j.kernel.impl.store.NeoStores;
import org.neo4j.kernel.impl.store.StoreFactory;
import org.neo4j.kernel.impl.store.format.RecordFormats;
import org.neo4j.kernel.impl.store.format.standard.Standard;
import org.neo4j.kernel.impl.store.record.NodeRecord;
import org.neo4j.kernel.impl.store.record.PropertyBlock;
import org.neo4j.logging.NullLogProvider;
import org.neo4j.test.extension.EphemeralNeo4jLayoutExtension;
import org.neo4j.test.extension.Inject;
import org.neo4j.test.extension.RandomExtension;
import org.neo4j.test.extension.pagecache.EphemeralPageCacheExtension;
import org.neo4j.test.rule.RandomRule;
import org.neo4j.values.storable.Value;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.neo4j.index.internal.gbptree.RecoveryCleanupWorkCollector.immediate;
import static org.neo4j.internal.helpers.collection.Iterators.iterator;
import static org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer.NULL;
import static org.neo4j.memory.EmptyMemoryTracker.INSTANCE;

@EphemeralPageCacheExtension
@EphemeralNeo4jLayoutExtension
@ExtendWith( RandomExtension.class )
public class RecordPropertyCursorTest
{
    @Inject
    protected RandomRule random;
    @Inject
    protected FileSystemAbstraction fs;
    @Inject
    protected PageCache pageCache;
    @Inject
    protected DatabaseLayout databaseLayout;

    protected NeoStores neoStores;
    protected PropertyCreator creator;
    protected NodeRecord owner;
    protected DefaultIdGeneratorFactory idGeneratorFactory;

    @BeforeEach
    void setup()
    {
        idGeneratorFactory = new DefaultIdGeneratorFactory( fs, immediate() );
        neoStores = new StoreFactory( databaseLayout, Config.defaults(), idGeneratorFactory,
                pageCache, fs, getRecordFormats(), NullLogProvider.getInstance(), PageCacheTracer.NULL, Sets.immutable.empty() ).openAllNeoStores( true );
        creator = new PropertyCreator( neoStores.getPropertyStore(), new PropertyTraverser( NULL ), NULL, INSTANCE );
        owner = neoStores.getNodeStore().newRecord();
    }

    protected RecordFormats getRecordFormats()
    {
        return Standard.LATEST_RECORD_FORMATS;
    }

    @AfterEach
    void closeStore()
    {
        neoStores.close();
    }

    @Test
    void shouldReadPropertyChain()
    {
        // given
        Value[] values = createValues();
        long firstPropertyId = storeValuesAsPropertyChain( creator, owner, values );

        // when
        assertPropertyChain( values, firstPropertyId, createCursor() );
    }

    @Test
    void shouldReuseCursor()
    {
        // given
        Value[] valuesA = createValues();
        long firstPropertyIdA = storeValuesAsPropertyChain( creator, owner, valuesA );
        Value[] valuesB = createValues();
        long firstPropertyIdB = storeValuesAsPropertyChain( creator, owner, valuesB );

        // then
        RecordPropertyCursor cursor = createCursor();
        assertPropertyChain( valuesA, firstPropertyIdA, cursor );
        assertPropertyChain( valuesB, firstPropertyIdB, cursor );
    }

    @Test
    void closeShouldBeIdempotent()
    {
        // given
        RecordPropertyCursor cursor = createCursor();

        // when
        cursor.close();

        // then
        cursor.close();
    }

    protected RecordPropertyCursor createCursor()
    {
        return new RecordPropertyCursor( neoStores.getPropertyStore(), NULL, INSTANCE );
    }

    protected static void assertPropertyChain( Value[] values, long firstPropertyId, RecordPropertyCursor cursor )
    {
        Map<Integer, Value> expectedValues = asMap( values );
        // This is a specific test for RecordPropertyCursor and we know that node/relationships init methods are the same
        cursor.initNodeProperties( firstPropertyId );
        while ( cursor.next() )
        {
            // then
            assertEquals( expectedValues.remove( cursor.propertyKey() ), cursor.propertyValue() );
        }
        assertTrue( expectedValues.isEmpty() );
    }

    protected Value[] createValues()
    {
        return createValues( 1, 20 );
    }

    protected Value[] createValues( int minNumProps, int maxNumProps )
    {
        int numberOfProperties = random.nextInt( minNumProps, maxNumProps );
        Value[] values = new Value[numberOfProperties];
        for ( int key = 0; key < numberOfProperties; key++ )
        {
            values[key] = random.nextValue();
        }
        return values;
    }

    protected long storeValuesAsPropertyChain( PropertyCreator creator, NodeRecord owner, Value[] values )
    {
        DirectRecordAccessSet access = new DirectRecordAccessSet( neoStores, idGeneratorFactory );
        long firstPropertyId = creator.createPropertyChain( owner, blocksOf( creator, values ), access.getPropertyRecords() );
        access.commit();
        return firstPropertyId;
    }

    protected static Map<Integer, Value> asMap( Value[] values )
    {
        Map<Integer, Value> map = new HashMap<>();
        for ( int key = 0; key < values.length; key++ )
        {
            map.put( key, values[key] );
        }
        return map;
    }

    protected static Iterator<PropertyBlock> blocksOf( PropertyCreator creator, Value[] values )
    {
        return new IteratorWrapper<>( iterator( values ) )
        {
            int key;

            @Override
            protected PropertyBlock underlyingObjectToObject( Value value )
            {
                return creator.encodePropertyValue( key++, value );
            }
        };
    }
}
