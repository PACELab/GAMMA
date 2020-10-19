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

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

import org.neo4j.exceptions.KernelException;
import org.neo4j.graphdb.Label;
import org.neo4j.graphdb.Transaction;
import org.neo4j.graphdb.schema.IndexCreator;
import org.neo4j.internal.helpers.collection.MapUtil;
import org.neo4j.internal.kernel.api.TokenWrite;
import org.neo4j.internal.kernel.api.exceptions.schema.DuplicateSchemaRuleException;
import org.neo4j.internal.kernel.api.exceptions.schema.SchemaRuleNotFoundException;
import org.neo4j.internal.recordstorage.RecordStorageEngine;
import org.neo4j.internal.recordstorage.SchemaStorage;
import org.neo4j.internal.schema.ConstraintDescriptor;
import org.neo4j.internal.schema.ConstraintType;
import org.neo4j.internal.schema.IndexConfig;
import org.neo4j.internal.schema.IndexDescriptor;
import org.neo4j.internal.schema.IndexPrototype;
import org.neo4j.internal.schema.LabelSchemaDescriptor;
import org.neo4j.internal.schema.SchemaDescriptor;
import org.neo4j.internal.schema.SchemaDescriptorPredicates;
import org.neo4j.internal.schema.SchemaRule;
import org.neo4j.internal.schema.constraints.ConstraintDescriptorFactory;
import org.neo4j.internal.schema.constraints.UniquenessConstraintDescriptor;
import org.neo4j.kernel.api.schema.index.TestIndexDescriptorFactory;
import org.neo4j.kernel.impl.coreapi.InternalTransaction;
import org.neo4j.kernel.impl.index.schema.GenericNativeIndexProvider;
import org.neo4j.kernel.internal.GraphDatabaseAPI;
import org.neo4j.test.extension.ImpermanentDbmsExtension;
import org.neo4j.test.extension.Inject;
import org.neo4j.token.TokenHolders;
import org.neo4j.values.storable.Values;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.neo4j.internal.helpers.ArrayUtil.single;
import static org.neo4j.internal.helpers.collection.Iterators.asSet;
import static org.neo4j.internal.schema.IndexPrototype.forSchema;
import static org.neo4j.internal.schema.IndexPrototype.uniqueForSchema;
import static org.neo4j.internal.schema.SchemaDescriptor.forLabel;
import static org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer.NULL;
import static org.neo4j.memory.EmptyMemoryTracker.INSTANCE;

@ImpermanentDbmsExtension
class SchemaStorageIT
{
    private static final String LABEL1 = "Label1";
    private static final String LABEL2 = "Label2";
    private static final String TYPE1 = "Type1";
    private static final String PROP1 = "prop1";
    private static final String PROP2 = "prop2";

    @Inject
    private GraphDatabaseAPI db;
    @Inject
    private RecordStorageEngine storageEngine;
    @Inject
    private TokenHolders tokenHolders;

    private static SchemaStore schemaStore;
    private static SchemaStorage storage;

    @BeforeEach
    void initStorage() throws Exception
    {
        try ( Transaction transaction = db.beginTx() )
        {
            TokenWrite tokenWrite = ((InternalTransaction) transaction).kernelTransaction().tokenWrite();
            tokenWrite.propertyKeyGetOrCreateForName( PROP1 );
            tokenWrite.propertyKeyGetOrCreateForName( PROP2 );
            tokenWrite.labelGetOrCreateForName( LABEL1 );
            tokenWrite.labelGetOrCreateForName( LABEL2 );
            tokenWrite.relationshipTypeGetOrCreateForName( TYPE1 );
            transaction.commit();
        }
        schemaStore = storageEngine.testAccessNeoStores().getSchemaStore();
        storage = new SchemaStorage( schemaStore, tokenHolders );
    }

    @Test
    void shouldReturnIndexRuleForLabelAndProperty()
    {
        // Given
        createSchema(
                index( LABEL1, PROP1 ),
                index( LABEL1, PROP2 ),
                index( LABEL2, PROP1 ) );

        // When
        IndexDescriptor rule = single( storage.indexGetForSchema( indexDescriptor( LABEL1, PROP2 ), NULL ) );

        // Then
        assertNotNull( rule );
        assertRule( rule, LABEL1, PROP2, false );
    }

    @Test
    void shouldReturnIndexRuleForLabelAndPropertyComposite()
    {
        String a = "a";
        String b = "b";
        String c = "c";
        String d = "d";
        String e = "e";
        String f = "f";
        createSchema( tx -> tx.schema().indexFor( Label.label( LABEL1 ) )
          .on( a ).on( b ).on( c ).on( d ).on( e ).on( f ).create() );

        IndexDescriptor rule = single( storage.indexGetForSchema( TestIndexDescriptorFactory.forLabel(
                labelId( LABEL1 ), propId( a ), propId( b ), propId( c ), propId( d ), propId( e ), propId( f ) ), NULL ) );

        assertNotNull( rule );
        assertTrue( SchemaDescriptorPredicates.hasLabel( rule, labelId( LABEL1 ) ) );
        assertTrue( SchemaDescriptorPredicates.hasProperty( rule, propId( a ) ) );
        assertTrue( SchemaDescriptorPredicates.hasProperty( rule, propId( b ) ) );
        assertTrue( SchemaDescriptorPredicates.hasProperty( rule, propId( c ) ) );
        assertTrue( SchemaDescriptorPredicates.hasProperty( rule, propId( d ) ) );
        assertTrue( SchemaDescriptorPredicates.hasProperty( rule, propId( e ) ) );
        assertTrue( SchemaDescriptorPredicates.hasProperty( rule, propId( f ) ) );
        assertFalse( rule.isUnique() );
    }

    @Test
    void shouldReturnIndexRuleForLabelAndVeryManyPropertiesComposite()
    {
        String[] props = "abcdefghijklmnopqrstuvwxyzABCDEFGHJILKMNOPQRSTUVWXYZ".split( "\\B" );
        createSchema( tx ->
        {
            IndexCreator indexCreator = tx.schema().indexFor( Label.label( LABEL1 ) );
            for ( String prop : props )
            {
                indexCreator = indexCreator.on( prop );
            }
            indexCreator.create();
        } );

        IndexDescriptor rule = single( storage.indexGetForSchema( TestIndexDescriptorFactory.forLabel(
                labelId( LABEL1 ), Arrays.stream( props ).mapToInt( this::propId ).toArray() ), NULL ) );

        assertNotNull( rule );
        assertTrue( SchemaDescriptorPredicates.hasLabel( rule, labelId( LABEL1 ) ) );
        for ( String prop : props )
        {
            assertTrue( SchemaDescriptorPredicates.hasProperty( rule, propId( prop ) ) );
        }
        assertFalse( rule.isUnique() );
    }

    @Test
    void shouldReturnEmptyArrayIfIndexRuleForLabelAndPropertyDoesNotExist()
    {
        // Given
        createSchema(
                index( LABEL1, PROP1 ) );

        // When
        IndexDescriptor[] rules = storage.indexGetForSchema( indexDescriptor( LABEL1, PROP2 ), NULL );

        // Then
        assertThat( rules.length ).isEqualTo( 0 );
    }

    @Test
    void shouldListIndexRulesForLabelPropertyAndKind()
    {
        // Given
        createSchema(
                uniquenessConstraint( LABEL1, PROP1 ),
                index( LABEL1, PROP2 ) );

        // When
        IndexDescriptor rule = single( storage.indexGetForSchema( uniqueIndexDescriptor( LABEL1, PROP1 ), NULL ) );

        // Then
        assertNotNull( rule );
        assertRule( rule, LABEL1, PROP1, true );
    }

    @Test
    void shouldListAllIndexRules()
    {
        // Given
        createSchema(
                index( LABEL1, PROP1 ),
                index( LABEL1, PROP2 ),
                uniquenessConstraint( LABEL2, PROP1 ) );

        // When
        Set<IndexDescriptor> listedRules = asSet( storage.indexesGetAll( NULL ) );

        // Then
        Set<IndexDescriptor> expectedRules = new HashSet<>();
        expectedRules.add( makeIndexRule( 1, LABEL1, PROP1 ) );
        expectedRules.add( makeIndexRule( 2, LABEL1, PROP2 ) );
        expectedRules.add( makeIndexRuleForConstraint( 3, LABEL2, PROP1, 0L ) );

        assertEquals( expectedRules, listedRules );
    }

    @Test
    void shouldReturnCorrectUniquenessRuleForLabelAndProperty()
            throws SchemaRuleNotFoundException, DuplicateSchemaRuleException
    {
        // Given
        createSchema(
                uniquenessConstraint( LABEL1, PROP1 ),
                uniquenessConstraint( LABEL2, PROP1 ) );

        // When
        ConstraintDescriptor rule = storage.constraintsGetSingle(
                ConstraintDescriptorFactory.uniqueForLabel( labelId( LABEL1 ), propId( PROP1 ) ), NULL );

        // Then
        assertNotNull( rule );
        assertRule( rule, LABEL1, PROP1, ConstraintType.UNIQUE );
    }

    @Test
    void shouldWriteAndReadIndexConfig() throws KernelException
    {
        // given
        IndexConfig expected = IndexConfig.with( MapUtil.genericMap(
                "value.string", Values.stringValue( "value" ),
                "value.int", Values.intValue( 1 ),
                "value.doubleArray", Values.doubleArray( new double[]{0.4, 0.6, 1.0} ),
                "value.boolean", Values.booleanValue( true )
        ) );
        var cursorTracer = NULL;
        SchemaDescriptor schema = forLabel( labelId( LABEL1 ), propId( PROP1 ) );
        long id = schemaStore.nextId( cursorTracer );
        IndexDescriptor storeIndexDescriptor = forSchema( schema ).withName( "index_" + id ).materialise( id ).withIndexConfig( expected );
        storage.writeSchemaRule( storeIndexDescriptor, cursorTracer, INSTANCE );

        // when
        IndexDescriptor schemaRule = (IndexDescriptor) storage.loadSingleSchemaRule( id, NULL );
        storage.deleteSchemaRule( schemaRule, NULL ); // Clean up after ourselves.

        // then
        IndexConfig actual = schemaRule.getIndexConfig();
        assertEquals( expected, actual, "Read index config not same as written, expected " + expected + ", actual " + actual );
    }

    private void assertRule( IndexDescriptor rule, String label, String propertyKey, boolean isUnique )
    {
        assertTrue( SchemaDescriptorPredicates.hasLabel( rule, labelId( label ) ) );
        assertTrue( SchemaDescriptorPredicates.hasProperty( rule, propId( propertyKey ) ) );
        assertEquals( isUnique, rule.isUnique() );
    }

    private void assertRule( ConstraintDescriptor constraint, String label, String propertyKey, ConstraintType type )
    {
        assertTrue( SchemaDescriptorPredicates.hasLabel( constraint, labelId( label ) ) );
        assertTrue( SchemaDescriptorPredicates.hasProperty( constraint, propId( propertyKey ) ) );
        assertEquals( type, constraint.type() );
    }

    private IndexDescriptor indexDescriptor( String label, String property )
    {
        return TestIndexDescriptorFactory.forLabel( labelId( label ), propId( property ) );
    }

    private IndexDescriptor uniqueIndexDescriptor( String label, String property )
    {
        return TestIndexDescriptorFactory.uniqueForLabel( labelId( label ), propId( property ) );
    }

    private IndexDescriptor makeIndexRule( long ruleId, String label, String propertyKey )
    {
        LabelSchemaDescriptor schema = forLabel( labelId( label ), propId( propertyKey ) );
        IndexPrototype prototype = forSchema( schema, GenericNativeIndexProvider.DESCRIPTOR );
        prototype = prototype.withName( SchemaRule.generateName( prototype, new String[]{label}, new String[]{propertyKey} ) );
        return prototype.materialise( ruleId );
    }

    private IndexDescriptor makeIndexRuleForConstraint( long ruleId, String label, String propertyKey, long constraintId )
    {
        LabelSchemaDescriptor schema = forLabel( labelId( label ), propId( propertyKey ) );
        IndexPrototype prototype = uniqueForSchema( schema, GenericNativeIndexProvider.DESCRIPTOR );
        UniquenessConstraintDescriptor constraint = ConstraintDescriptorFactory.uniqueForSchema( schema );
        prototype = prototype.withName( SchemaRule.generateName( constraint, new String[]{label}, new String[]{propertyKey} ) );
        return prototype.materialise( ruleId ).withOwningConstraintId( constraintId );
    }

    private int labelId( String labelName )
    {
        try ( Transaction tx = db.beginTx() )
        {
            return ((InternalTransaction) tx).kernelTransaction().tokenRead().nodeLabel( labelName );
        }
    }

    private int propId( String propName )
    {
        try ( Transaction tx = db.beginTx() )
        {
            return ((InternalTransaction) tx).kernelTransaction().tokenRead().propertyKey( propName );
        }
    }

    private static Consumer<Transaction> index( String label, String prop )
    {
        return tx -> tx.schema().indexFor( Label.label( label ) ).on( prop ).create();
    }

    private static Consumer<Transaction> uniquenessConstraint( String label, String prop )
    {
        return tx -> tx.schema().constraintFor( Label.label( label ) ).assertPropertyIsUnique( prop ).create();
    }

    @SafeVarargs
    private void createSchema( Consumer<Transaction>... creators )
    {
        try ( Transaction tx = db.beginTx() )
        {
            for ( Consumer<Transaction> rule : creators )
            {
                rule.accept( tx );
            }
            tx.commit();
        }
        awaitIndexes();
    }

    private void awaitIndexes()
    {
        try ( Transaction tx = db.beginTx() )
        {
            tx.schema().awaitIndexesOnline( 1, TimeUnit.MINUTES );
            tx.commit();
        }
    }
}
