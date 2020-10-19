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
package org.neo4j.importer;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.parallel.ResourceLock;
import org.junit.jupiter.api.parallel.Resources;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.Collections;
import java.util.List;

import org.neo4j.configuration.Config;
import org.neo4j.configuration.GraphDatabaseSettings;
import org.neo4j.csv.reader.Configuration;
import org.neo4j.io.layout.DatabaseLayout;
import org.neo4j.io.pagecache.tracing.DefaultPageCacheTracer;
import org.neo4j.test.extension.Inject;
import org.neo4j.test.extension.Neo4jLayoutExtension;
import org.neo4j.test.extension.SuppressOutputExtension;
import org.neo4j.test.rule.TestDirectory;

import static java.util.Collections.emptySet;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Neo4jLayoutExtension
@ExtendWith( SuppressOutputExtension.class )
@ResourceLock( Resources.SYSTEM_OUT )
class CsvImporterTest
{
    @Inject
    private TestDirectory testDir;
    @Inject
    private DatabaseLayout databaseLayout;

    @Test
    void writesReportToSpecifiedReportFile() throws Exception
    {

        File logDir = testDir.directory( "logs" );
        File reportLocation = testDir.file( "the_report" );

        File inputFile = testDir.file( "foobar.csv" );
        List<String> lines = Collections.singletonList( "foo\\tbar\\tbaz" );
        Files.write( inputFile.toPath(), lines, Charset.defaultCharset() );

        Config config = Config.defaults( GraphDatabaseSettings.logs_directory, logDir.toPath().toAbsolutePath() );

        CsvImporter csvImporter = CsvImporter.builder()
            .withDatabaseLayout( databaseLayout )
            .withDatabaseConfig( config )
            .withReportFile( reportLocation.getAbsoluteFile() )
            .withCsvConfig( Configuration.TABS )
            .withFileSystem( testDir.getFileSystem() )
            .addNodeFiles( emptySet(), new File[]{inputFile.getAbsoluteFile()} )
            .build();

        csvImporter.doImport();

        assertTrue( reportLocation.exists() );
    }

    @Test
    void tracePageCacheAccessOnCsvImport() throws IOException
    {
        File logDir = testDir.directory( "logs" );
        File reportLocation = testDir.file( "the_report" );
        File inputFile = testDir.file( "foobar.csv" );

        List<String> lines = List.of( "foo;bar;baz" );
        Files.write( inputFile.toPath(), lines, Charset.defaultCharset() );

        Config config = Config.defaults( GraphDatabaseSettings.logs_directory, logDir.toPath().toAbsolutePath() );

        var cacheTracer = new DefaultPageCacheTracer();
        CsvImporter csvImporter = CsvImporter.builder()
                .withDatabaseLayout( databaseLayout )
                .withDatabaseConfig( config )
                .withReportFile( reportLocation.getAbsoluteFile() )
                .withFileSystem( testDir.getFileSystem() )
                .withPageCacheTracer( cacheTracer )
                .addNodeFiles( emptySet(), new File[]{inputFile.getAbsoluteFile()} )
                .build();

        csvImporter.doImport();

        assertThat( cacheTracer.faults() ).isEqualTo( 24 );
        assertThat( cacheTracer.pins() ).isEqualTo( 414 );
        assertThat( cacheTracer.unpins() ).isEqualTo( 414 );
        assertThat( cacheTracer.hits() ).isEqualTo( 390 );
    }
}
