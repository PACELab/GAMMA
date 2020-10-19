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
package org.neo4j.kernel.impl.storemigration;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;

import org.neo4j.common.ProgressReporter;
import org.neo4j.internal.schema.IndexProviderDescriptor;
import org.neo4j.io.fs.FileSystemAbstraction;
import org.neo4j.io.layout.DatabaseLayout;
import org.neo4j.io.layout.Neo4jLayout;
import org.neo4j.kernel.api.index.IndexDirectoryStructure;
import org.neo4j.kernel.api.index.IndexProvider;
import org.neo4j.storageengine.api.StorageEngineFactory;
import org.neo4j.storageengine.api.StoreVersion;
import org.neo4j.storageengine.api.format.CapabilityType;
import org.neo4j.storageengine.migration.SchemaIndexMigrator;
import org.neo4j.test.extension.Inject;
import org.neo4j.test.extension.testdirectory.TestDirectoryExtension;
import org.neo4j.test.rule.TestDirectory;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.neo4j.configuration.GraphDatabaseSettings.DEFAULT_DATABASE_NAME;

@TestDirectoryExtension
class SchemaIndexMigratorTest
{
    @Inject
    private TestDirectory testDirectory;

    private final FileSystemAbstraction fs = mock( FileSystemAbstraction.class );
    private final File home = mock( File.class );
    private final ProgressReporter progressReporter = mock( ProgressReporter.class );
    private final IndexProvider indexProvider = mock( IndexProvider.class );
    private DatabaseLayout databaseLayout;
    private DatabaseLayout migrationLayout;

    @BeforeEach
    void setup()
    {
        databaseLayout = Neo4jLayout.of( testDirectory.directoryPath( "store" ) ).databaseLayout( DEFAULT_DATABASE_NAME );
        migrationLayout = Neo4jLayout.of( testDirectory.directoryPath( "migrationDir" ) ).databaseLayout( DEFAULT_DATABASE_NAME );
    }

    @Test
    void schemaAndLabelIndexesRemovedAfterSuccessfulMigration() throws IOException
    {
        StorageEngineFactory storageEngineFactory = mock( StorageEngineFactory.class );
        StoreVersion version = mock( StoreVersion.class );
        when( version.hasCompatibleCapabilities( any(), eq( CapabilityType.INDEX ) ) ).thenReturn( false );
        when( storageEngineFactory.versionInformation( anyString() ) ).thenReturn( version );
        IndexDirectoryStructure directoryStructure = mock( IndexDirectoryStructure.class );
        Path indexProviderRootDirectory = databaseLayout.file( "just-some-directory" );
        when( directoryStructure.rootDirectory() ).thenReturn( indexProviderRootDirectory );
        SchemaIndexMigrator migrator = new SchemaIndexMigrator( "Test migrator", fs, directoryStructure, storageEngineFactory );
        when( indexProvider.getProviderDescriptor() )
                .thenReturn( new IndexProviderDescriptor( "key", "version" ) );

        migrator.migrate( databaseLayout, migrationLayout, progressReporter, "from", "to" );
        migrator.moveMigratedFiles( migrationLayout, databaseLayout, "from", "to" );

        verify( fs ).deleteRecursively( indexProviderRootDirectory.toFile() );
    }
}
