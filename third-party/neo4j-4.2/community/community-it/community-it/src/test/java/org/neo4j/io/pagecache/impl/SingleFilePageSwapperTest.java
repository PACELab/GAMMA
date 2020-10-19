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
package org.neo4j.io.pagecache.impl;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.junit.jupiter.api.function.ThrowingSupplier;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.channels.FileLock;
import java.nio.channels.OverlappingFileLockException;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import org.neo4j.adversaries.RandomAdversary;
import org.neo4j.adversaries.fs.AdversarialFileSystemAbstraction;
import org.neo4j.internal.helpers.NamedThreadFactory;
import org.neo4j.internal.unsafe.UnsafeUtil;
import org.neo4j.io.IOUtils;
import org.neo4j.io.fs.DefaultFileSystemAbstraction;
import org.neo4j.io.fs.DelegatingFileSystemAbstraction;
import org.neo4j.io.fs.DelegatingStoreChannel;
import org.neo4j.io.fs.EphemeralFileSystemAbstraction;
import org.neo4j.io.fs.FileSystemAbstraction;
import org.neo4j.io.fs.StoreChannel;
import org.neo4j.io.fs.StoreFileChannel;
import org.neo4j.io.memory.ByteBuffers;
import org.neo4j.io.pagecache.PageSwapper;
import org.neo4j.io.pagecache.PageSwapperFactory;
import org.neo4j.io.pagecache.PageSwapperTest;

import static java.util.concurrent.ConcurrentHashMap.newKeySet;
import static java.util.concurrent.TimeUnit.MILLISECONDS;
import static java.util.concurrent.locks.LockSupport.parkNanos;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.neo4j.memory.EmptyMemoryTracker.INSTANCE;
import static org.neo4j.test.proc.ProcessUtil.getClassPath;
import static org.neo4j.test.proc.ProcessUtil.getJavaExecutable;

public class SingleFilePageSwapperTest extends PageSwapperTest
{
    private EphemeralFileSystemAbstraction ephemeralFileSystem;
    private DefaultFileSystemAbstraction fileSystem;
    private Path path;
    private ExecutorService operationExecutor;
    private ThreadRegistryFactory threadRegistryFactory;
    private static final int INTERRUPT_ATTEMPTS = 100;

    @BeforeEach
    void setUp() throws IOException
    {
        path = Path.of( "file" ).normalize();
        ephemeralFileSystem = new EphemeralFileSystemAbstraction();
        fileSystem = new DefaultFileSystemAbstraction();
        threadRegistryFactory = new ThreadRegistryFactory();
        operationExecutor = Executors.newSingleThreadExecutor( threadRegistryFactory );
    }

    @AfterEach
    void tearDown() throws Exception
    {
        operationExecutor.shutdown();
        IOUtils.closeAll( ephemeralFileSystem, fileSystem );
    }

    @Override
    protected PageSwapperFactory swapperFactory( FileSystemAbstraction fileSystem )
    {
        return new SingleFilePageSwapperFactory( fileSystem );
    }

    @Override
    protected void mkdirs( Path dir ) throws IOException
    {
        getFs().mkdirs( dir.toFile() );
    }

    protected Path getPath()
    {
        return path;
    }

    @Override
    protected FileSystemAbstraction getFs()
    {
        return getEphemeralFileSystem();
    }

    private FileSystemAbstraction getEphemeralFileSystem()
    {
        return ephemeralFileSystem;
    }

    FileSystemAbstraction getRealFileSystem()
    {
        return fileSystem;
    }

    private void putBytes( long page, byte[] data, int srcOffset, int tgtOffset, int length )
    {
        for ( int i = 0; i < length; i++ )
        {
            UnsafeUtil.putByte( page + srcOffset + i, data[tgtOffset + i] );
        }
    }

    @Test
    void swappingInMustFillPageWithData() throws Exception
    {
        byte[] bytes = new byte[] { 1, 2, 3, 4 };
        StoreChannel channel = getFs().write( getPath().toFile() );
        channel.writeAll( wrap( bytes ) );
        channel.close();

        PageSwapperFactory factory = createSwapperFactory( getFs() );
        PageSwapper swapper = createSwapper( factory, getPath(), 4, null, false );
        long target = createPage( 4 );
        assertEquals( 4, swapper.read( 0, target ) );

        assertThat( array( target ) ).containsExactly( bytes );
    }

    @Test
    void mustZeroFillPageBeyondEndOfFile() throws Exception
    {
        byte[] bytes = new byte[] {
                // --- page 0:
                1, 2, 3, 4,
                // --- page 1:
                5, 6
        };
        StoreChannel channel = getFs().write( getPath().toFile() );
        channel.writeAll( wrap( bytes ) );
        channel.close();

        PageSwapperFactory factory = createSwapperFactory( getFs() );
        PageSwapper swapper = createSwapper( factory, getPath(), 4, null, false );
        long target = createPage( 4 );
        assertEquals( 2, swapper.read( 1, target ) );

        assertThat( array( target ) ).containsExactly( 5, 6, 0, 0 );
    }

    @Test
    void uninterruptibleRead() throws Exception
    {
        byte[] pageContent = new byte[] {1, 2, 3, 4};
        StoreChannel channel = getFs().write( getPath().toFile() );
        channel.writeAll( wrap( pageContent ) );
        channel.close();

        PageSwapperFactory factory = createSwapperFactory( getFs() );
        PageSwapper swapper = createSwapper( factory, getPath(), 4, null, false );
        long target = createPage( 4 );

        CountDownLatch startInterruptsLatch = new CountDownLatch( 1 );
        AtomicBoolean readFlag = new AtomicBoolean( true );
        var uninterruptibleFuture = operationExecutor.submit( () ->
        {
            startInterruptsLatch.countDown();
            while ( readFlag.get() )
            {
                try
                {
                    swapper.read( 0, target );
                }
                catch ( Throwable t )
                {
                    throw new RuntimeException( t );
                }
            }
        } );

        startInterruptsLatch.await();
        assertFalse( threadRegistryFactory.getThreads().isEmpty() );

        for ( int i = 0; i < INTERRUPT_ATTEMPTS; i++ )
        {
            threadRegistryFactory.getThreads().forEach( Thread::interrupt );
            parkNanos( MILLISECONDS.toNanos( 10 ) );
        }
        readFlag.set( false );
        assertDoesNotThrow( (ThrowingSupplier<?>) uninterruptibleFuture::get );
    }

    @Test
    void uninterruptibleWrite() throws Exception
    {
        byte[] pageContent = new byte[] {1, 2, 3, 4};
        StoreChannel channel = getFs().write( getPath().toFile() );
        channel.writeAll( wrap( pageContent ) );
        channel.close();

        PageSwapperFactory factory = createSwapperFactory( getFs() );
        PageSwapper swapper = createSwapper( factory, getPath(), 4, null, false );
        long target = createPage( 4 );

        CountDownLatch startInterruptsLatch = new CountDownLatch( 1 );
        AtomicBoolean writeFlag = new AtomicBoolean( true );
        var uninterruptibleFuture = operationExecutor.submit( () ->
        {
            startInterruptsLatch.countDown();
            while ( writeFlag.get() )
            {
                try
                {
                    swapper.write( 0, target );
                }
                catch ( Throwable t )
                {
                    throw new RuntimeException( t );
                }
            }
        } );

        startInterruptsLatch.await();
        assertFalse( threadRegistryFactory.getThreads().isEmpty() );

        for ( int i = 0; i < INTERRUPT_ATTEMPTS; i++ )
        {
            threadRegistryFactory.getThreads().forEach( Thread::interrupt );
            parkNanos( MILLISECONDS.toNanos( 10 ) );
        }
        writeFlag.set( false );
        assertDoesNotThrow( (ThrowingSupplier<?>) uninterruptibleFuture::get );
    }

    @Test
    void swappingOutMustWritePageToFile() throws Exception
    {
        getFs().write( getPath().toFile() ).close();

        byte[] expected = new byte[] { 1, 2, 3, 4 };
        long page = createPage( expected );

        PageSwapperFactory factory = createSwapperFactory( getFs() );
        PageSwapper swapper = createSwapper( factory, getPath(), 4, null, false );
        assertEquals( 4, swapper.write( 0, page ) );

        try ( InputStream stream = getFs().openAsInputStream( getPath().toFile() ) )
        {
            byte[] actual = new byte[expected.length];

            assertThat( stream.read( actual ) ).isEqualTo( actual.length );
            assertThat( actual ).containsExactly( expected );
        }
    }

    private long createPage( byte[] expected )
    {
        long page = createPage( expected.length );
        putBytes( page, expected, 0, 0, expected.length );
        return page;
    }

    @Test
    void swappingOutMustNotOverwriteDataBeyondPage() throws Exception
    {
        byte[] initialData = new byte[] {
                // --- page 0:
                1, 2, 3, 4,
                // --- page 1:
                5, 6, 7, 8,
                // --- page 2:
                9, 10
        };
        byte[] finalData = new byte[] {
                // --- page 0:
                1, 2, 3, 4,
                // --- page 1:
                8, 7, 6, 5,
                // --- page 2:
                9, 10
        };
        StoreChannel channel = getFs().write( getPath().toFile() );
        channel.writeAll( wrap( initialData ) );
        channel.close();

        byte[] change = new byte[] { 8, 7, 6, 5 };
        long page = createPage( change );

        PageSwapperFactory factory = createSwapperFactory( getFs() );
        PageSwapper swapper = createSwapper( factory, getPath(), 4, null, false );
        swapper.write( 1, page );

        try ( InputStream stream = getFs().openAsInputStream( getPath().toFile() ) )
        {
            byte[] actual = new byte[(int) getFs().getFileSize( getPath().toFile() )];

            assertThat( stream.read( actual ) ).isEqualTo( actual.length );
            assertThat( actual ).containsExactly( finalData );
        }
    }

    /**
     * The OverlappingFileLockException is thrown when tryLock is called on the same file *in the same JVM*.
     */
    @Test
    @DisabledOnOs( OS.WINDOWS )
    void creatingSwapperForFileMustTakeLockOnFile() throws Exception
    {
        PageSwapperFactory factory = createSwapperFactory( fileSystem );
        Path file = testDir.filePath( "file" );
        fileSystem.write( file.toFile() ).close();

        PageSwapper pageSwapper = createSwapper( factory, file, 4, NO_CALLBACK, false );

        try
        {
            StoreChannel channel = fileSystem.write( file.toFile() );
            assertThrows( OverlappingFileLockException.class, channel::tryLock );
        }
        finally
        {
            pageSwapper.close();
        }
    }

    @Test
    @DisabledOnOs( OS.WINDOWS )
    void creatingSwapperForInternallyLockedFileMustThrow() throws Exception
    {
        PageSwapperFactory factory = createSwapperFactory( fileSystem );
        Path file = testDir.filePath( "file" );

        StoreFileChannel channel = fileSystem.write( file.toFile() );

        try ( FileLock fileLock = channel.tryLock() )
        {
            assertThat( fileLock ).isNotNull();
            assertThrows( FileLockException.class, () -> createSwapper( factory, file, 4, NO_CALLBACK, true ) );
        }
    }

    @Test
    @DisabledOnOs( OS.WINDOWS )
    void creatingSwapperForExternallyLockedFileMustThrow() throws Exception
    {
        PageSwapperFactory factory = createSwapperFactory( fileSystem );
        Path file = testDir.filePath( "file" );

        fileSystem.write( file.toFile() ).close();

        ProcessBuilder pb = new ProcessBuilder(
                getJavaExecutable().toString(),
                "-cp", getClassPath(),
                LockThisFileProgram.class.getCanonicalName(), file.toAbsolutePath().toString() );
        File wd = new File( "target/test-classes" ).getAbsoluteFile();
        pb.directory( wd );
        Process process = pb.start();
        BufferedReader stdout = new BufferedReader( new InputStreamReader( process.getInputStream() ) );
        InputStream stderr = process.getErrorStream();
        try
        {
            assumeTrue( LockThisFileProgram.LOCKED_OUTPUT.equals( stdout.readLine() ) );
        }
        catch ( Throwable e )
        {
            int b = stderr.read();
            while ( b != -1 )
            {
                System.err.write( b );
                b = stderr.read();
            }
            System.err.flush();
            int exitCode = process.waitFor();
            System.out.println( "exitCode = " + exitCode );
            throw e;
        }

        try
        {
            assertThrows( FileLockException.class, () -> createSwapper( factory, file, 4, NO_CALLBACK, true ) );
        }
        finally
        {
            process.getOutputStream().write( 0 );
            process.getOutputStream().flush();
            process.waitFor();
        }
    }

    @Test
    @DisabledOnOs( OS.WINDOWS )
    void mustUnlockFileWhenThePageSwapperIsClosed() throws Exception
    {
        PageSwapperFactory factory = createSwapperFactory( fileSystem );
        Path file = testDir.filePath( "file" );
        fileSystem.write( file.toFile() ).close();

        createSwapper( factory, file, 4, NO_CALLBACK, false ).close();

        try ( StoreFileChannel channel = fileSystem.write( file.toFile() );
              FileLock fileLock = channel.tryLock() )
        {
            assertThat( fileLock ).isNotNull();
        }
    }

    @Test
    @DisabledOnOs( OS.WINDOWS )
    void fileMustRemainLockedEvenIfChannelIsClosedByStrayInterrupt() throws Exception
    {
        PageSwapperFactory factory = createSwapperFactory( fileSystem );
        Path file = testDir.filePath( "file" );
        fileSystem.write( file.toFile() ).close();

        PageSwapper pageSwapper = createSwapper( factory, file, 4, NO_CALLBACK, false );

        try
        {
            StoreChannel channel = fileSystem.write( file.toFile() );

            Thread.currentThread().interrupt();
            pageSwapper.force();

            assertThrows( OverlappingFileLockException.class, channel::tryLock );
        }
        finally
        {
            pageSwapper.close();
        }
    }

    @Test
    @DisabledOnOs( OS.WINDOWS )
    void mustCloseFilesIfTakingFileLockThrows() throws Exception
    {
        final AtomicInteger openFilesCounter = new AtomicInteger();
        PageSwapperFactory factory = createSwapperFactory( new DelegatingFileSystemAbstraction( fileSystem )
        {
            @Override
            public StoreChannel open( File fileName, Set<OpenOption> options ) throws IOException
            {
                openFilesCounter.getAndIncrement();
                return new DelegatingStoreChannel( super.open( fileName, options ) )
                {
                    @Override
                    public void close() throws IOException
                    {
                        openFilesCounter.getAndDecrement();
                        super.close();
                    }
                };
            }
        } );
        Path file = testDir.filePath( "file" );
        try ( StoreChannel ch = fileSystem.write( file.toFile() );
                FileLock ignore = ch.tryLock() )
        {
            createSwapper( factory, file, 4, NO_CALLBACK, false ).close();
            fail( "Creating a page swapper for a locked channel should have thrown" );
        }
        catch ( FileLockException e )
        {
            // As expected.
        }
        assertThat( openFilesCounter.get() ).isEqualTo( 0 );
    }

    private byte[] array( long page )
    {
        int size = sizeOfAsInt( page );
        byte[] array = new byte[size];
        for ( int i = 0; i < size; i++ )
        {
            array[i] = UnsafeUtil.getByte( page + i );
        }
        return array;
    }

    private ByteBuffer wrap( byte[] bytes )
    {
        ByteBuffer buffer = ByteBuffers.allocate( bytes.length, INSTANCE );
        for ( byte b : bytes )
        {
            buffer.put( b );
        }
        buffer.clear();
        return buffer;
    }

    @Test
    void mustHandleMischiefInPositionedRead() throws Exception
    {
        int bytesTotal = 512;
        byte[] data = new byte[bytesTotal];
        ThreadLocalRandom.current().nextBytes( data );

        PageSwapperFactory factory = createSwapperFactory( getFs() );
        Path file = getPath();
        PageSwapper swapper = createSwapper( factory, file, bytesTotal, NO_CALLBACK, true );
        try
        {
            long page = createPage( data );
            swapper.write( 0, page );
        }
        finally
        {
            swapper.close();
            factory.close();
        }

        RandomAdversary adversary = new RandomAdversary( 0.5, 0.0, 0.0 );
        factory = createSwapperFactory( new AdversarialFileSystemAbstraction( adversary, getFs() ) );
        swapper = createSwapper( factory, file, bytesTotal, NO_CALLBACK, false );

        long page = createPage( bytesTotal );

        try
        {
            for ( int i = 0; i < 10_000; i++ )
            {
                clear( page );
                assertThat( swapper.read( 0, page ) ).isEqualTo( bytesTotal );
                assertThat( array( page ) ).isEqualTo( data );
            }
        }
        finally
        {
            swapper.close();
        }
    }

    @Test
    void mustHandleMischiefInPositionedWrite() throws Exception
    {
        int bytesTotal = 512;
        byte[] data = new byte[bytesTotal];
        ThreadLocalRandom.current().nextBytes( data );
        long zeroPage = createPage( bytesTotal );
        clear( zeroPage );

        Path file = getPath();
        RandomAdversary adversary = new RandomAdversary( 0.5, 0.0, 0.0 );
        PageSwapperFactory factory = createSwapperFactory( new AdversarialFileSystemAbstraction( adversary, getFs() ) );
        PageSwapper swapper = createSwapper( factory, file, bytesTotal, NO_CALLBACK, true );

        long page = createPage( bytesTotal );

        try
        {
            for ( int i = 0; i < 10_000; i++ )
            {
                adversary.setProbabilityFactor( 0 );
                swapper.write( 0, zeroPage );
                putBytes( page, data, 0, 0, data.length );
                adversary.setProbabilityFactor( 1 );
                assertThat( swapper.write( 0, page ) ).isEqualTo( bytesTotal );
                clear( page );
                adversary.setProbabilityFactor( 0 );
                swapper.read( 0, page );
                assertThat( array( page ) ).isEqualTo( data );
            }
        }
        finally
        {
            swapper.close();
        }
    }

    @Test
    void mustHandleMischiefInPositionedVectoredRead() throws Exception
    {
        int bytesTotal = 512;
        int bytesPerPage = 32;
        int pageCount = bytesTotal / bytesPerPage;
        byte[] data = new byte[bytesTotal];
        ThreadLocalRandom.current().nextBytes( data );

        PageSwapperFactory factory = createSwapperFactory( getFs() );
        Path file = getPath();
        PageSwapper swapper = createSwapper( factory, file, bytesTotal, NO_CALLBACK, true );
        try
        {
            long page = createPage( data );
            swapper.write( 0, page );
        }
        finally
        {
            swapper.close();
        }

        RandomAdversary adversary = new RandomAdversary( 0.5, 0.0, 0.0 );
        factory = createSwapperFactory( new AdversarialFileSystemAbstraction( adversary, getFs() ) );
        swapper = createSwapper( factory, file, bytesPerPage, NO_CALLBACK, false );

        long[] pages = new long[pageCount];
        int[] pageLengths = new int[pageCount];
        for ( int i = 0; i < pageCount; i++ )
        {
            pages[i] = createPage( bytesPerPage );
            pageLengths[i] = bytesPerPage;
        }

        byte[] temp = new byte[bytesPerPage];
        try
        {
            for ( int i = 0; i < 10_000; i++ )
            {
                for ( long page : pages )
                {
                    clear( page );
                }
                assertThat( swapper.read( 0, pages, pageLengths, pages.length ) ).isEqualTo( bytesTotal );
                for ( int j = 0; j < pageCount; j++ )
                {
                    System.arraycopy( data, j * bytesPerPage, temp, 0, bytesPerPage );
                    assertThat( array( pages[j] ) ).isEqualTo( temp );
                }
            }
        }
        finally
        {
            swapper.close();
        }
    }

    @Test
    void mustHandleMischiefInPositionedVectoredWrite() throws Exception
    {
        int bytesTotal = 512;
        int bytesPerPage = 32;
        int pageCount = bytesTotal / bytesPerPage;
        byte[] data = new byte[bytesTotal];
        ThreadLocalRandom.current().nextBytes( data );
        long zeroPage = createPage( bytesPerPage );
        clear( zeroPage );

        Path file = getPath();
        RandomAdversary adversary = new RandomAdversary( 0.5, 0.0, 0.0 );
        PageSwapperFactory factory = createSwapperFactory( new AdversarialFileSystemAbstraction( adversary, getFs() ) );
        PageSwapper swapper = createSwapper( factory, file, bytesPerPage, NO_CALLBACK, true );

        long[] writePages = new long[pageCount];
        long[] readPages = new long[pageCount];
        long[] zeroPages = new long[pageCount];
        int[] pageLengths = new int[pageCount];
        for ( int i = 0; i < pageCount; i++ )
        {
            writePages[i] = createPage( bytesPerPage );
            putBytes( writePages[i], data, 0, i * bytesPerPage, bytesPerPage );
            readPages[i] = createPage( bytesPerPage );
            zeroPages[i] = zeroPage;
            pageLengths[i] = bytesPerPage;
        }

        try
        {
            for ( int i = 0; i < 10_000; i++ )
            {
                adversary.setProbabilityFactor( 0 );
                swapper.write( 0, zeroPages, pageLengths, pageCount, pageCount );
                adversary.setProbabilityFactor( 1 );
                swapper.write( 0, writePages, pageLengths, pageCount, pageCount );
                for ( long readPage : readPages )
                {
                    clear( readPage );
                }
                adversary.setProbabilityFactor( 0 );
                assertThat( swapper.read( 0, readPages, pageLengths, pageCount ) ).isEqualTo( bytesTotal );
                for ( int j = 0; j < pageCount; j++ )
                {
                    assertThat( array( readPages[j] ) ).containsExactly( array( writePages[j] ) );
                }
            }
        }
        finally
        {
            swapper.close();
        }
    }

    private static class ThreadRegistryFactory extends NamedThreadFactory
    {
        private final Set<Thread> threads = newKeySet();

        ThreadRegistryFactory()
        {
            super( "SwapperInterruptTestThreads" );
        }

        @Override
        public Thread newThread( Runnable runnable )
        {
            var thread = super.newThread( runnable );
            threads.add( thread );
            return thread;
        }

        public Set<Thread> getThreads()
        {
            return threads;
        }
    }
}
