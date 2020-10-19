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
package org.neo4j.io.fs;

import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;

import org.neo4j.test.extension.Inject;
import org.neo4j.test.rule.TestDirectory;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

abstract class FileSystemUtilsTest
{
    @Inject
    private FileSystemAbstraction fs;
    @Inject
    private TestDirectory testDirectory;

    @Test
    void shouldCheckNonExistingDirectory()
    {
        File nonExistingDir = new File( "nonExistingDir" );

        assertTrue( FileSystemUtils.isEmptyOrNonExistingDirectory( fs, nonExistingDir ) );
    }

    @Test
    void shouldCheckExistingEmptyDirectory()
    {
        File existingEmptyDir = testDirectory.directory( "existingEmptyDir" );

        assertTrue( FileSystemUtils.isEmptyOrNonExistingDirectory( fs, existingEmptyDir ) );
    }

    @Test
    void dropDirectoryWithFile() throws IOException
    {
        File directory = testDirectory.directory( "directory" );
        fs.openAsOutputStream( new File( directory, "a" ), false ).close();

        assertEquals( 1, fs.listFiles( directory ).length );

        FileSystemUtils.deleteFile( fs, directory );

        assertThat( fs.listFiles( directory ) ).isNullOrEmpty();
    }

    @Test
    void shouldCheckExistingNonEmptyDirectory() throws Exception
    {
        File existingEmptyDir = testDirectory.directory( "existingEmptyDir" );
        fs.write( new File( existingEmptyDir, "someFile" ) ).close();

        assertFalse( FileSystemUtils.isEmptyOrNonExistingDirectory( fs, existingEmptyDir ) );
    }

    @Test
    void shouldCheckExistingFile()
    {
        File existingFile = testDirectory.createFile( "existingFile" );

        assertFalse( FileSystemUtils.isEmptyOrNonExistingDirectory( fs, existingFile ) );
    }

    @Test
    void shouldCheckSizeOfFile() throws Exception
    {
        File file = testDirectory.createFile( "a" );

        try ( var fileWriter = fs.openAsWriter( file, UTF_8, false ) )
        {
            fileWriter.append( 'a' );
        }

        assertThat( FileSystemUtils.size( fs, file ) ).isEqualTo( 1L );
    }

    @Test
    void shouldCheckSizeOfDirectory() throws Exception
    {
        File dir = testDirectory.directory( "dir" );
        File file1 = new File( dir, "file1" );
        File file2 = new File( dir, "file2" );

        try ( var fileWriter = fs.openAsWriter( file1, UTF_8, false ) )
        {
            fileWriter.append( 'a' ).append( 'b' );
        }
        try ( var fileWriter = fs.openAsWriter( file2, UTF_8, false ) )
        {
            fileWriter.append( 'a' );
        }

        assertThat( FileSystemUtils.size( fs, dir ) ).isEqualTo( 3L );
    }
}
