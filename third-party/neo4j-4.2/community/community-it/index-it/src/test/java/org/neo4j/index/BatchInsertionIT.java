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
package org.neo4j.index;

import org.junit.Rule;
import org.junit.Test;

import java.io.IOException;
import java.util.Collections;

import org.neo4j.batchinsert.BatchInserter;
import org.neo4j.batchinsert.BatchInserters;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Transaction;
import org.neo4j.internal.id.IdValidator;
import org.neo4j.internal.id.ReservedIdException;
import org.neo4j.test.rule.EmbeddedDbmsRule;
import org.neo4j.test.rule.fs.DefaultFileSystemRule;

import static org.assertj.core.api.Assertions.assertThat;
import static org.neo4j.graphdb.Label.label;
import static org.neo4j.internal.helpers.collection.Iterators.count;
import static org.neo4j.internal.helpers.collection.MapUtil.map;

public class BatchInsertionIT
{
    @Rule
    public final EmbeddedDbmsRule dbRule = new EmbeddedDbmsRule().startLazily();
    @Rule
    public final DefaultFileSystemRule fileSystemRule = new DefaultFileSystemRule();

    @Test
    public void shouldIndexNodesWithMultipleLabels() throws Exception
    {
        // Given
        BatchInserter inserter = BatchInserters.inserter( dbRule.databaseLayout(), fileSystemRule.get() );

        inserter.createNode( map( "name", "Bob" ), label( "User" ), label( "Admin" ) );

        inserter.createDeferredSchemaIndex( label( "User" ) ).on( "name" ).create();
        inserter.createDeferredSchemaIndex( label( "Admin" ) ).on( "name" ).create();

        // When
        inserter.shutdown();

        // Then
        GraphDatabaseService db = dbRule.getGraphDatabaseAPI();
        try ( Transaction tx = db.beginTx() )
        {
            assertThat( count( tx.findNodes( label( "User" ), "name", "Bob" ) ) ).isEqualTo( 1L );
            assertThat( count( tx.findNodes( label( "Admin" ), "name", "Bob" ) ) ).isEqualTo( 1L );
        }
    }

    @Test
    public void shouldNotIndexNodesWithWrongLabel() throws Exception
    {
        // Given
        BatchInserter inserter = BatchInserters.inserter( dbRule.databaseLayout(), fileSystemRule.get() );

        inserter.createNode( map("name", "Bob"), label( "User" ), label("Admin"));

        inserter.createDeferredSchemaIndex( label( "Banana" ) ).on( "name" ).create();

        // When
        inserter.shutdown();

        // Then
        GraphDatabaseService db = dbRule.getGraphDatabaseAPI();
        try ( Transaction tx = db.beginTx() )
        {
            assertThat( count( tx.findNodes( label( "Banana" ), "name", "Bob" ) ) ).isEqualTo( 0L );
        }

    }

    @Test
    public void shouldBeAbleToMakeRepeatedCallsToSetNodeProperty() throws Exception
    {
        BatchInserter inserter = BatchInserters.inserter( dbRule.databaseLayout(), fileSystemRule.get() );
        long nodeId = inserter.createNode( Collections.emptyMap() );

        final Object finalValue = 87;
        inserter.setNodeProperty( nodeId, "a", "some property value" );
        inserter.setNodeProperty( nodeId, "a", 42 );
        inserter.setNodeProperty( nodeId, "a", 3.14 );
        inserter.setNodeProperty( nodeId, "a", true );
        inserter.setNodeProperty( nodeId, "a", finalValue );
        inserter.shutdown();

        GraphDatabaseService db = dbRule.getGraphDatabaseAPI();
        try ( Transaction tx = db.beginTx() )
        {
            assertThat( tx.getNodeById( nodeId ).getProperty( "a" ) ).isEqualTo( finalValue );
        }
    }

    @Test
    public void shouldBeAbleToMakeRepeatedCallsToSetNodePropertyWithMultiplePropertiesPerBlock() throws Exception
    {
        BatchInserter inserter = BatchInserters.inserter( dbRule.databaseLayout(), fileSystemRule.get() );
        long nodeId = inserter.createNode( Collections.emptyMap() );

        final Object finalValue1 = 87;
        final Object finalValue2 = 3.14;
        inserter.setNodeProperty( nodeId, "a", "some property value" );
        inserter.setNodeProperty( nodeId, "a", 42 );
        inserter.setNodeProperty( nodeId, "b", finalValue2 );
        inserter.setNodeProperty( nodeId, "a", finalValue2 );
        inserter.setNodeProperty( nodeId, "a", true );
        inserter.setNodeProperty( nodeId, "a", finalValue1 );
        inserter.shutdown();

        GraphDatabaseService db = dbRule.getGraphDatabaseAPI();
        try ( Transaction tx = db.beginTx() )
        {
            assertThat( tx.getNodeById( nodeId ).getProperty( "a" ) ).isEqualTo( finalValue1 );
            assertThat( tx.getNodeById( nodeId ).getProperty( "b" ) ).isEqualTo( finalValue2 );
        }
    }

    @Test( expected = ReservedIdException.class )
    public void makeSureCantCreateNodeWithMagicNumber() throws IOException
    {
        // given
        BatchInserter inserter = BatchInserters.inserter( dbRule.databaseLayout(), fileSystemRule.get() );
        try
        {
            // when
            long id = IdValidator.INTEGER_MINUS_ONE;
            inserter.createNode( id, null );
            // then throws
        }
        finally
        {
            inserter.shutdown();
        }
    }
}
