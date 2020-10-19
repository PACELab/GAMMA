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
package org.neo4j.io.layout;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collection;

import org.neo4j.configuration.Config;
import org.neo4j.test.extension.Inject;
import org.neo4j.test.extension.testdirectory.TestDirectoryExtension;
import org.neo4j.test.rule.TestDirectory;

import static java.util.stream.Collectors.toSet;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.neo4j.configuration.GraphDatabaseInternalSettings.databases_root_path;
import static org.neo4j.configuration.GraphDatabaseSettings.DEFAULT_DATABASES_ROOT_DIR_NAME;
import static org.neo4j.configuration.GraphDatabaseSettings.DEFAULT_DATA_DIR_NAME;
import static org.neo4j.internal.helpers.collection.Iterators.asSet;

@TestDirectoryExtension
class Neo4jLayoutTest
{
    @Inject
    private TestDirectory testDirectory;

    @Test
    void storeLayoutForAbsoluteFile()
    {
        Path storeDir = testDirectory.directoryPath( "store" );
        Neo4jLayout storeLayout = Neo4jLayout.of( Config.defaults( databases_root_path, storeDir ) );
        assertEquals( storeDir, storeLayout.databasesDirectory() );
    }

    @Test
    void storeLayoutResolvesLinks() throws IOException
    {
        Path basePath = testDirectory.homePath();
        Path storeDir = testDirectory.homePath("notAbsolute");
        Path linkPath = basePath.resolve( "link" );
        Path symbolicLink = Files.createSymbolicLink( linkPath, storeDir );
        Neo4jLayout storeLayout = Neo4jLayout.of( Config.defaults( databases_root_path, symbolicLink ) );
        assertEquals( storeDir, storeLayout.databasesDirectory() );
    }

    @Test
    void storeLayoutUseCanonicalRepresentation()
    {
        Path basePath = testDirectory.homeDir("notCanonical").toPath();
        Path notCanonicalPath = basePath.resolve( "../anotherLocation" );
        Neo4jLayout storeLayout = Neo4jLayout.of( notCanonicalPath );
        assertEquals( testDirectory.directoryPath( "anotherLocation" ), storeLayout.homeDirectory() );
    }

    @Test
    void storeLockFileLocation()
    {
        Neo4jLayout layout = Neo4jLayout.of( testDirectory.homePath() );
        Path storeLockFile = layout.storeLockFile();
        assertEquals( "store_lock", storeLockFile.getFileName().toString() );
        assertEquals( layout.databasesDirectory(), storeLockFile.getParent() );
    }

    @Test
    void serverIdFileLocation()
    {
        Neo4jLayout layout = Neo4jLayout.of( testDirectory.homePath() );
        Path serverIdFile = layout.serverIdFile();
        assertEquals( "server_id", serverIdFile.getFileName().toString() );
        assertEquals( testDirectory.directoryPath( DEFAULT_DATA_DIR_NAME ), serverIdFile.getParent() );
    }

    @Test
    void emptyStoreLayoutDatabasesCollection()
    {
        Neo4jLayout storeLayout = Neo4jLayout.of( testDirectory.homePath() );
        assertTrue( storeLayout.databaseLayouts().isEmpty() );
    }

    @Test
    void storeLayoutDatabasesOnlyBasedOnSubfolders()
    {
        Neo4jLayout layout = Neo4jLayout.of( testDirectory.homePath() );

        testDirectory.directory( "abc", DEFAULT_DATA_DIR_NAME, DEFAULT_DATABASES_ROOT_DIR_NAME );
        testDirectory.directory( "bcd", DEFAULT_DATA_DIR_NAME, DEFAULT_DATABASES_ROOT_DIR_NAME );
        testDirectory.createFile( "cde", DEFAULT_DATA_DIR_NAME, DEFAULT_DATABASES_ROOT_DIR_NAME );

        Collection<DatabaseLayout> layouts = layout.databaseLayouts();
        assertEquals( 2, layouts.size() );
        assertEquals( asSet( "abc", "bcd" ), layouts.stream().map( DatabaseLayout::getDatabaseName ).collect( toSet() ) );
    }
}
