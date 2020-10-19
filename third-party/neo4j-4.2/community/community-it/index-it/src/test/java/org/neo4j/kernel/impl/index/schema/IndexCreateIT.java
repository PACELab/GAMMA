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

import org.junit.jupiter.api.Test;

import org.neo4j.common.EntityType;
import org.neo4j.configuration.GraphDatabaseSettings;
import org.neo4j.exceptions.KernelException;
import org.neo4j.internal.kernel.api.SchemaWrite;
import org.neo4j.internal.schema.IndexDescriptor;
import org.neo4j.internal.schema.FulltextSchemaDescriptor;
import org.neo4j.internal.schema.IndexPrototype;
import org.neo4j.internal.schema.LabelSchemaDescriptor;
import org.neo4j.internal.schema.SchemaDescriptor;
import org.neo4j.kernel.api.exceptions.schema.RepeatedLabelInSchemaException;
import org.neo4j.kernel.api.exceptions.schema.RepeatedPropertyInSchemaException;
import org.neo4j.kernel.api.exceptions.schema.RepeatedRelationshipTypeInSchemaException;
import org.neo4j.kernel.impl.api.index.IndexProviderNotFoundException;
import org.neo4j.kernel.impl.api.integrationtest.KernelIntegrationTest;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.neo4j.internal.schema.SchemaDescriptor.forLabel;

public class IndexCreateIT extends KernelIntegrationTest
{
    private static final IndexCreator INDEX_CREATOR = SchemaWrite::indexCreate;
    private static final IndexCreator UNIQUE_CONSTRAINT_CREATOR = ( schemaWrite, schema, provider, name ) -> schemaWrite.uniquePropertyConstraintCreate(
            IndexPrototype.uniqueForSchema( schema, schemaWrite.indexProviderByName( provider ) ).withName( name ) );

    @Test
    void shouldCreateIndexWithSpecificExistingProviderName() throws KernelException
    {
        shouldCreateWithSpecificExistingProviderName( INDEX_CREATOR );
    }

    @Test
    void shouldCreateUniquePropertyConstraintWithSpecificExistingProviderName() throws KernelException
    {
        shouldCreateWithSpecificExistingProviderName( UNIQUE_CONSTRAINT_CREATOR );
    }

    @Test
    void shouldFailCreateIndexWithNonExistentProviderName() throws KernelException
    {
        shouldFailWithNonExistentProviderName( INDEX_CREATOR );
    }

    @Test
    void shouldFailCreateUniquePropertyConstraintWithNonExistentProviderName() throws KernelException
    {
        shouldFailWithNonExistentProviderName( UNIQUE_CONSTRAINT_CREATOR );
    }

    @Test
    void shouldFailCreateIndexWithDuplicateLabels() throws KernelException
    {
        // given
        SchemaWrite schemaWrite = schemaWriteInNewTransaction();

        // when
        final FulltextSchemaDescriptor descriptor = SchemaDescriptor.fulltext( EntityType.NODE, new int[]{0, 0}, new int[]{1} );
        // then
        assertThrows( RepeatedLabelInSchemaException.class, () -> schemaWrite.indexCreate( descriptor, null ) );
    }

    @Test
    void shouldFailCreateIndexWithDuplicateRelationshipTypes() throws KernelException
    {
        // given
        SchemaWrite schemaWrite = schemaWriteInNewTransaction();

        // when
        final FulltextSchemaDescriptor descriptor = SchemaDescriptor.fulltext( EntityType.RELATIONSHIP, new int[]{0, 0}, new int[]{1} );
        // then
        assertThrows( RepeatedRelationshipTypeInSchemaException.class, () -> schemaWrite.indexCreate( descriptor, null ) );
    }

    @Test
    void shouldFailCreateIndexWithDuplicateProperties() throws KernelException
    {
        // given
        SchemaWrite schemaWrite = schemaWriteInNewTransaction();

        // when
        final FulltextSchemaDescriptor descriptor = SchemaDescriptor.fulltext( EntityType.NODE, new int[]{0}, new int[]{1, 1} );
        // then
        assertThrows( RepeatedPropertyInSchemaException.class, () -> schemaWrite.indexCreate( descriptor, null ) );
    }

    protected void shouldFailWithNonExistentProviderName( IndexCreator creator ) throws KernelException
    {
        // given
        SchemaWrite schemaWrite = schemaWriteInNewTransaction();

        // when
        assertThrows( IndexProviderNotFoundException.class,
            () -> creator.create( schemaWrite, forLabel( 0, 0 ), "something-completely-different", "index name" ) );
    }

    protected void shouldCreateWithSpecificExistingProviderName( IndexCreator creator ) throws KernelException
    {
        int labelId = 0;
        for ( GraphDatabaseSettings.SchemaIndex indexSetting : GraphDatabaseSettings.SchemaIndex.values() )
        {
            // given
            SchemaWrite schemaWrite = schemaWriteInNewTransaction();
            String provider = indexSetting.providerName();
            LabelSchemaDescriptor descriptor = forLabel( labelId++, 0 );
            String indexName = "index-" + labelId;
            creator.create( schemaWrite, descriptor, provider, indexName );
            IndexDescriptor index = transaction.kernelTransaction().schemaRead().indexGetForName( indexName );

            // when
            commit();

            // then
            assertEquals( provider, indexingService.getIndexProxy( index ).getDescriptor().getIndexProvider().name() );
        }
    }

    protected interface IndexCreator
    {
        void create( SchemaWrite schemaWrite, LabelSchemaDescriptor descriptor, String providerName, String indexName ) throws KernelException;
    }
}
