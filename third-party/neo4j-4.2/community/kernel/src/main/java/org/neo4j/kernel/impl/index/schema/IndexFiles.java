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

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;

import org.neo4j.io.compress.ZipUtils;
import org.neo4j.io.fs.FileSystemAbstraction;
import org.neo4j.kernel.api.index.IndexDirectoryStructure;

/**
 * Surface for a schema indexes to act on the files that it owns.
 * One instance of this class maps to a single index or sub-index if living under a Fusion umbrella.
 * Wraps all {@link IOException IOExceptions} in {@link UncheckedIOException}.
 */
public class IndexFiles
{
    private final FileSystemAbstraction fs;
    private final Path directory;
    private final Path storeFile;

    public IndexFiles( FileSystemAbstraction fs, IndexDirectoryStructure directoryStructure, long indexId )
    {
        this.fs = fs;
        this.directory = directoryStructure.directoryForIndex( indexId );
        this.storeFile = directory.resolve( indexFileName( indexId ) );
    }

    private static String indexFileName( long indexId )
    {
        return "index-" + indexId;
    }

    public Path getStoreFile()
    {
        return storeFile;
    }

    public Path getBase()
    {
        return directory;
    }

    public void clear()
    {
        try
        {
            fs.deleteRecursively( directory.toFile() );
        }
        catch ( IOException e )
        {
            throw new UncheckedIOException( e );
        }
    }

    public void archiveIndex()
    {
        if ( fs.isDirectory( directory.toFile() ) && fs.fileExists( directory.toFile() ) && fs.listFiles( directory.toFile() ).length > 0 )
        {
            try
            {
                ZipUtils.zip( fs, directory,
                        directory.getParent().resolve( "archive-" + directory.getFileName() + "-" + System.currentTimeMillis() + ".zip" ) );
            }
            catch ( IOException e )
            {
                throw new UncheckedIOException( e );
            }
        }
    }

    public void ensureDirectoryExist()
    {
        try
        {
            fs.mkdirs( directory.toFile() );
        }
        catch ( IOException e )
        {
            throw new UncheckedIOException( e );
        }
    }

    @Override
    public String toString()
    {
        return String.format( "%s[base=%s,storeFile=%s]", getClass().getSimpleName(), getBase(), getStoreFile() );
    }
}
