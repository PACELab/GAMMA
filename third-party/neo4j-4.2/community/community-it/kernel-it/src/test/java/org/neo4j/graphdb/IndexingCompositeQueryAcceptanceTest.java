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
package org.neo4j.graphdb;

import org.eclipse.collections.api.iterator.LongIterator;
import org.eclipse.collections.api.set.primitive.LongSet;
import org.eclipse.collections.api.set.primitive.MutableLongSet;
import org.eclipse.collections.impl.set.mutable.primitive.LongHashSet;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

import org.neo4j.graphdb.schema.IndexCreator;
import org.neo4j.kernel.internal.GraphDatabaseAPI;
import org.neo4j.test.extension.ImpermanentDbmsExtension;
import org.neo4j.test.extension.Inject;

import static org.assertj.core.api.Assertions.assertThat;
import static org.neo4j.internal.helpers.collection.Iterators.array;

@ImpermanentDbmsExtension
public class IndexingCompositeQueryAcceptanceTest
{
    @Inject
    private GraphDatabaseAPI db;

    public static Stream<Arguments> data()
    {
        return Stream.of(
                testCase( array( 2, 3 ), biIndexSeek, true ),
                testCase( array( 2, 3 ), biIndexSeek, false ),
                testCase( array( 2, 3, 4 ), triIndexSeek, true ),
                testCase( array( 2, 3, 4 ), triIndexSeek, false ),
                testCase( array( 2, 3, 4, 5, 6 ), mapIndexSeek, true ),
                testCase( array( 2, 3, 4, 5, 6 ), mapIndexSeek, false )
        );
    }

    private static final Label LABEL = Label.label( "LABEL1" );

    public void setup( boolean withIndex, String[] keys )
    {
        if ( withIndex )
        {
            try ( org.neo4j.graphdb.Transaction tx = db.beginTx() )
            {
                tx.schema().indexFor( LABEL ).on( keys[0] ).create();

                IndexCreator indexCreator = tx.schema().indexFor( LABEL );
                for ( String key : keys )
                {
                    indexCreator = indexCreator.on( key );
                }
                indexCreator.create();
                tx.commit();
            }

            try ( org.neo4j.graphdb.Transaction tx = db.beginTx() )
            {
                tx.schema().awaitIndexesOnline( 5, TimeUnit.MINUTES );
                tx.commit();
            }
        }
    }

    @ParameterizedTest
    @MethodSource( "data" )
    public void shouldSupportIndexSeek( String[] keys, Object[] values, Object[][] nonMatching, IndexSeek indexSeek, boolean withIndex )
    {
        setup( withIndex, keys );

        // GIVEN
        createNodes( db, LABEL, keys, nonMatching );
        LongSet expected = createNodes( db, LABEL, keys, values );

        // WHEN
        MutableLongSet found = new LongHashSet();
        try ( Transaction tx = db.beginTx() )
        {
            collectNodes( found, indexSeek.findNodes( keys, values, db, tx ) );
        }

        // THEN
        assertThat( found ).isEqualTo( expected );
    }

    @ParameterizedTest
    @MethodSource( "data" )
    public void shouldSupportIndexSeekBackwardsOrder( String[] keys, Object[] values, Object[][] nonMatching, IndexSeek indexSeek, boolean withIndex )
    {
        setup( withIndex, keys );

        // GIVEN
        createNodes( db, LABEL, keys, nonMatching );
        LongSet expected = createNodes( db, LABEL, keys, values );

        // WHEN
        MutableLongSet found = new LongHashSet();
        String[] reversedKeys = new String[keys.length];
        Object[] reversedValues = new Object[keys.length];
        for ( int i = 0; i < keys.length; i++ )
        {
            reversedValues[keys.length - 1 - i] = values[i];
            reversedKeys[keys.length - 1 - i] = keys[i];
        }
        try ( Transaction tx = db.beginTx() )
        {
            collectNodes( found, indexSeek.findNodes( reversedKeys, reversedValues, db, tx ) );
        }

        // THEN
        assertThat( found ).isEqualTo( expected );
    }

    @ParameterizedTest
    @MethodSource( "data" )
    public void shouldIncludeNodesCreatedInSameTxInIndexSeek( String[] keys, Object[] values, Object[][] nonMatching, IndexSeek indexSeek, boolean withIndex )
    {
        setup( withIndex, keys );

        // GIVEN
        createNodes( db, LABEL, keys, nonMatching[0], nonMatching[1] );
        MutableLongSet expected = createNodes( db, LABEL, keys, values );
        // WHEN
        MutableLongSet found = new LongHashSet();
        try ( Transaction tx = db.beginTx() )
        {
            expected.add( createNode( tx, propertyMap( keys, values ), LABEL ).getId() );
            createNode( tx, propertyMap( keys, nonMatching[2] ), LABEL );

            collectNodes( found, indexSeek.findNodes( keys, values, db, tx ) );
        }
        // THEN
        assertThat( found ).isEqualTo( expected );
    }

    @ParameterizedTest
    @MethodSource( "data" )
    public void shouldNotIncludeNodesDeletedInSameTxInIndexSeek( String[] keys, Object[] values, Object[][] nonMatching, IndexSeek indexSeek,
            boolean withIndex )
    {
        setup( withIndex, keys );

        // GIVEN
        createNodes( db, LABEL, keys, nonMatching[0] );
        LongSet toDelete = createNodes( db, LABEL, keys, values, nonMatching[1], nonMatching[2] );
        MutableLongSet expected = createNodes( db, LABEL, keys, values );
        // WHEN
        MutableLongSet found = new LongHashSet();
        try ( Transaction tx = db.beginTx() )
        {
            LongIterator deleting = toDelete.longIterator();
            while ( deleting.hasNext() )
            {
                long id = deleting.next();
                tx.getNodeById( id ).delete();
                expected.remove( id );
            }

            collectNodes( found, indexSeek.findNodes( keys, values, db, tx ) );
        }
        // THEN
        assertThat( found ).isEqualTo( expected );
    }

    @ParameterizedTest
    @MethodSource( "data" )
    public void shouldConsiderNodesChangedInSameTxInIndexSeek( String[] keys, Object[] values, Object[][] nonMatching, IndexSeek indexSeek, boolean withIndex )
    {
        setup( withIndex, keys );

        // GIVEN
        createNodes( db, LABEL, keys, nonMatching[0] );
        LongSet toChangeToMatch = createNodes( db, LABEL, keys, nonMatching[1] );
        LongSet toChangeToNotMatch = createNodes( db, LABEL, keys, values );
        MutableLongSet expected = createNodes( db, LABEL, keys, values );
        // WHEN
        MutableLongSet found = new LongHashSet();
        try ( Transaction tx = db.beginTx() )
        {
            LongIterator toMatching = toChangeToMatch.longIterator();
            while ( toMatching.hasNext() )
            {
                long id = toMatching.next();
                setProperties( tx, id, keys, values );
                expected.add( id );
            }
            LongIterator toNotMatching = toChangeToNotMatch.longIterator();
            while ( toNotMatching.hasNext() )
            {
                long id = toNotMatching.next();
                setProperties( tx, id, keys, nonMatching[2] );
                expected.remove( id );
            }

            collectNodes( found, indexSeek.findNodes( keys, values, db, tx ) );
        }
        // THEN
        assertThat( found ).isEqualTo( expected );
    }

    private MutableLongSet createNodes( GraphDatabaseService db, Label label, String[] keys, Object[]... propertyValueTuples )
    {
        MutableLongSet expected = new LongHashSet();
        try ( Transaction tx = db.beginTx() )
        {
            for ( Object[] valueTuple : propertyValueTuples )
            {
                expected.add( createNode( tx, propertyMap( keys, valueTuple ), label ).getId() );
            }
            tx.commit();
        }
        return expected;
    }

    private static Map<String,Object> propertyMap( String[] keys, Object[] valueTuple )
    {
        Map<String,Object> propertyValues = new HashMap<>();
        for ( int i = 0; i < keys.length; i++ )
        {
            propertyValues.put( keys[i], valueTuple[i] );
        }
        return propertyValues;
    }

    private void collectNodes( MutableLongSet bucket, ResourceIterator<Node> toCollect )
    {
        while ( toCollect.hasNext() )
        {
            bucket.add( toCollect.next().getId() );
        }
    }

    public Node createNode( Transaction transaction, Map<String, Object> properties, Label... labels )
    {
        Node node = transaction.createNode( labels );
        for ( Map.Entry<String,Object> property : properties.entrySet() )
        {
            node.setProperty( property.getKey(), property.getValue() );
        }
        return node;
    }

    private static Arguments testCase( Integer[] values, IndexSeek indexSeek, boolean withIndex )
    {
        Object[][] nonMatching = {plus( values, 1 ), plus( values, 2 ), plus( values, 3 )};
        String[] keys = Arrays.stream( values ).map( v -> "key" + v ).toArray( String[]::new );
        return Arguments.of( keys, values, nonMatching, indexSeek, withIndex );
    }

    private static <T> Object[] plus( Integer[] values, int offset )
    {
        Object[] result = new Object[values.length];
        for ( int i = 0; i < values.length; i++ )
        {
            result[i] = values[i] + offset;
        }
        return result;
    }

    private void setProperties( Transaction tx, long id, String[] keys, Object[] values )
    {
        Node node = tx.getNodeById( id );
        for ( int i = 0; i < keys.length; i++ )
        {
            node.setProperty( keys[i], values[i] );
        }
    }

    private interface IndexSeek
    {
        ResourceIterator<Node> findNodes( String[] keys, Object[] values, GraphDatabaseService db, Transaction tx );
    }

    private static final IndexSeek biIndexSeek =
            ( keys, values, db, tx ) ->
            {
                assert keys.length == 2;
                assert values.length == 2;
                return tx.findNodes( LABEL, keys[0], values[0], keys[1], values[1] );
            };

    private static final IndexSeek triIndexSeek =
            ( keys, values, db, tx ) ->
            {
                assert keys.length == 3;
                assert values.length == 3;
                return tx.findNodes( LABEL, keys[0], values[0], keys[1], values[1], keys[2], values[2] );
            };

    private static final IndexSeek mapIndexSeek =
            ( keys, values, db, tx ) -> tx.findNodes( LABEL, propertyMap( keys, values ) );
}
