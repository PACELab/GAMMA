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
package org.neo4j.kernel.impl.transaction.log.files;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicLong;

import org.neo4j.io.ByteUnit;
import org.neo4j.io.IOUtils;
import org.neo4j.io.fs.DelegatingStoreChannel;
import org.neo4j.io.fs.FileSystemAbstraction;
import org.neo4j.io.memory.NativeScopedBuffer;
import org.neo4j.io.pagecache.tracing.PageCacheTracer;
import org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer;
import org.neo4j.kernel.impl.transaction.log.FlushablePositionAwareChecksumChannel;
import org.neo4j.kernel.impl.transaction.log.LogPosition;
import org.neo4j.kernel.impl.transaction.log.LogVersionBridge;
import org.neo4j.kernel.impl.transaction.log.LogVersionedStoreChannel;
import org.neo4j.kernel.impl.transaction.log.PhysicalLogVersionedStoreChannel;
import org.neo4j.kernel.impl.transaction.log.PositionAwarePhysicalFlushableChecksumChannel;
import org.neo4j.kernel.impl.transaction.log.ReadAheadLogChannel;
import org.neo4j.kernel.impl.transaction.log.ReadableLogChannel;
import org.neo4j.kernel.impl.transaction.log.ReaderLogVersionBridge;
import org.neo4j.kernel.impl.transaction.log.entry.LogEntry;
import org.neo4j.kernel.impl.transaction.log.entry.LogEntryReader;
import org.neo4j.kernel.lifecycle.LifecycleAdapter;
import org.neo4j.memory.MemoryTracker;
import org.neo4j.storageengine.api.LogVersionRepository;

import static java.lang.Math.min;
import static java.lang.Runtime.getRuntime;

/**
 * {@link LogFile} backed by one or more files in a {@link FileSystemAbstraction}.
 */
class TransactionLogFile extends LifecycleAdapter implements LogFile
{
    private static final String TRANSACTION_LOG_FILE_ROTATION_TAG = "transactionLogFileRotation";
    private final AtomicLong rotateAtSize;
    private final LogFiles logFiles;
    private final TransactionLogFilesContext context;
    private final LogVersionBridge readerLogVersionBridge;
    private final PageCacheTracer pageCacheTracer;
    private final MemoryTracker memoryTracker;

    private volatile PhysicalLogVersionedStoreChannel channel;
    private PositionAwarePhysicalFlushableChecksumChannel writer;
    private LogVersionRepository logVersionRepository;

    TransactionLogFile( LogFiles logFiles, TransactionLogFilesContext context )
    {
        this.rotateAtSize = context.getRotationThreshold();
        this.context = context;
        this.logFiles = logFiles;
        this.readerLogVersionBridge = new ReaderLogVersionBridge( logFiles );
        this.pageCacheTracer = context.getDatabaseTracers().getPageCacheTracer();
        memoryTracker = context.getMemoryTracker();
    }

    @Override
    public void init() throws IOException
    {
        logVersionRepository = context.getLogVersionRepository();
    }

    @Override
    public void start() throws IOException
    {
        long currentLogVersion = logVersionRepository.getCurrentLogVersion();
        channel = logFiles.createLogChannelForVersion( currentLogVersion, context::getLastCommittedTransactionId );

        //try to set position
        seekChannelPosition( currentLogVersion );

        writer = new PositionAwarePhysicalFlushableChecksumChannel( channel, new NativeScopedBuffer( calculateLogBufferSize(), memoryTracker ) );
    }

    private void seekChannelPosition( long currentLogVersion ) throws IOException
    {
        jumpToTheLastClosedTxPosition( currentLogVersion );
        LogPosition position;
        try
        {
            position = scanToEndOfLastLogEntry();
        }
        catch ( Exception e )
        {
            // If we can't read the log, it could be that the last-closed-transaction position in the meta-data store is wrong.
            // We can try again by scanning the log file from the start.
            jumpToLogStart( currentLogVersion );
            try
            {
                position = scanToEndOfLastLogEntry();
            }
            catch ( Exception exception )
            {
                exception.addSuppressed( e );
                throw exception;
            }
        }
        channel.position( position.getByteOffset() );
    }

    private LogPosition scanToEndOfLastLogEntry() throws IOException
    {
        // scroll all over possible checkpoints
        try ( ReadAheadLogChannel readAheadLogChannel = new ReadAheadLogChannel( new UncloseableChannel( channel ), memoryTracker ) )
        {
            LogEntryReader logEntryReader = context.getLogEntryReader();
            LogEntry entry;
            do
            {
                // seek to the end the records.
                entry = logEntryReader.readLogEntry( readAheadLogChannel );
            }
            while ( entry != null );
            return logEntryReader.lastPosition();
        }
    }

    private void jumpToTheLastClosedTxPosition( long currentLogVersion ) throws IOException
    {
        LogPosition logPosition = context.getLastClosedTransactionPosition();
        long lastTxOffset = logPosition.getByteOffset();
        long lastTxLogVersion = logPosition.getLogVersion();
        long headerSize = logFiles.extractHeader( currentLogVersion ).getStartPosition().getByteOffset();
        if ( lastTxOffset < headerSize || channel.size() < lastTxOffset )
        {
            return;
        }
        if ( lastTxLogVersion == currentLogVersion )
        {
            channel.position( lastTxOffset );
        }
    }

    private void jumpToLogStart( long currentLogVersion ) throws IOException
    {
        long headerSize = logFiles.extractHeader( currentLogVersion ).getStartPosition().getByteOffset();
        channel.position( headerSize );
    }

    // In order to be able to write into a logfile after life.stop during shutdown sequence
    // we will close channel and writer only during shutdown phase when all pending changes (like last
    // checkpoint) are already in
    @Override
    public void shutdown() throws IOException
    {
        IOUtils.closeAll( writer );
    }

    @Override
    public boolean rotationNeeded()
    {
        /*
         * Whereas channel.size() should be fine, we're safer calling position() due to possibility
         * of this file being memory mapped or whatever.
         */
        return channel.position() >= rotateAtSize.get();
    }

    @Override
    public synchronized File rotate() throws IOException
    {
        try ( var cursorTracer = pageCacheTracer.createPageCursorTracer( TRANSACTION_LOG_FILE_ROTATION_TAG ) )
        {
            channel = rotate( channel, cursorTracer );
            writer.setChannel( channel );
            return channel.getFile();
        }
    }

    /**
     * Rotates the current log file, continuing into next (version) log file.
     * This method must be recovery safe, which means a crash at any point should be recoverable.
     * Concurrent readers must also be able to parry for concurrent rotation.
     * Concurrent writes will not be an issue since rotation and writing contends on the same monitor.
     *
     * Steps during rotation are:
     * <ol>
     * <li>1: Increment log version, {@link LogVersionRepository#incrementAndGetVersion(PageCursorTracer)} (also flushes the store)</li>
     * <li>2: Flush current log</li>
     * <li>3: Create new log file</li>
     * <li>4: Write header</li>
     * </ol>
     *
     * Recovery: what happens if crash between:
     * <ol>
     * <li>1-2: New log version has been set, starting the writer will create the new log file idempotently.
     * At this point there may have been half-written transactions in the previous log version,
     * although they haven't been considered committed and so they will be truncated from log during recovery</li>
     * <li>2-3: New log version has been set, starting the writer will create the new log file idempotently.
     * At this point there may be complete transactions in the previous log version which may not have been
     * acknowledged to be committed back to the user, but will be considered committed anyway.</li>
     * <li>3-4: New log version has been set, starting the writer will see that the new file exists and
     * will be forgiving when trying to read the header of it, so that if it isn't complete a fresh
     * header will be set.</li>
     * </ol>
     *
     * Reading: what happens when rotation is between:
     * <ol>
     * <li>1-2: Reader bridge will see that there's a new version (when asking {@link LogVersionRepository}
     * and try to open it. The log file doesn't exist yet though. The bridge can parry for this by catching
     * {@link FileNotFoundException} and tell the reader that the stream has ended</li>
     * <li>2-3: Same as (1-2)</li>
     * <li>3-4: Here the new log file exists, but the header may not be fully written yet.
     * the reader will fail when trying to read the header since it's reading it strictly and bridge
     * catches that exception, treating it the same as if the file didn't exist.</li>
     * </ol>
     *
     * @param currentLog current {@link LogVersionedStoreChannel channel} to flush and close.
     * @param cursorTracer underlying page cursor tracer.
     * @return the channel of the newly opened/created log file.
     * @throws IOException if an error regarding closing or opening log files occur.
     */
    private PhysicalLogVersionedStoreChannel rotate( LogVersionedStoreChannel currentLog, PageCursorTracer cursorTracer ) throws IOException
    {
        /*
         * The store is now flushed. If we fail now the recovery code will open the
         * current log file and replay everything. That's unnecessary but totally ok.
         */
        long newLogVersion = logVersionRepository.incrementAndGetVersion( cursorTracer );
        /*
         * Rotation can happen at any point, although not concurrently with an append,
         * although an append may have (most likely actually) left at least some bytes left
         * in the buffer for future flushing. Flushing that buffer now makes the last appended
         * transaction complete in the log we're rotating away. Awesome.
         */
        writer.prepareForFlush().flush();
        currentLog.truncate( currentLog.position() );

        /*
         * The log version is now in the store, flushed and persistent. If we crash
         * now, on recovery we'll attempt to open the version we're about to create
         * (but haven't yet), discover it's not there. That will lead to creating
         * the file, setting the header and continuing.
         * We using committing transaction id as a source of last transaction id here since
         * we can have transactions that are not yet published as committed but were already stored
         * into transaction log that was just rotated.
         */
        PhysicalLogVersionedStoreChannel newLog = logFiles.createLogChannelForVersion( newLogVersion, context::committingTransactionId );
        currentLog.close();
        return newLog;
    }

    @Override
    public FlushablePositionAwareChecksumChannel getWriter()
    {
        return writer;
    }

    @Override
    public ReadableLogChannel getReader( LogPosition position ) throws IOException
    {
        return getReader( position, readerLogVersionBridge );
    }

    @Override
    public ReadableLogChannel getReader( LogPosition position, LogVersionBridge logVersionBridge ) throws IOException
    {
        PhysicalLogVersionedStoreChannel logChannel = logFiles.openForVersion( position.getLogVersion() );
        logChannel.position( position.getByteOffset() );
        return new ReadAheadLogChannel( logChannel, logVersionBridge, memoryTracker );
    }

    @Override
    public void accept( LogFileVisitor visitor, LogPosition startingFromPosition ) throws IOException
    {
        try ( ReadableLogChannel reader = getReader( startingFromPosition ) )
        {
            visitor.visit( reader );
        }
    }

    /**
     * Calculate size of byte buffer for transaction log file based on number of available cpu's.
     * Minimal buffer size is 512KB. Every another 4 cpu's will add another 512KB into the buffer size.
     * Maximal buffer size is 4MB taking into account that we can have more then one transaction log writer in multi-database env.
     * <p/>
     * Examples:
     * runtime with 4 cpus will have buffer size of 1MB
     * runtime with 8 cpus will have buffer size of 1MB 512KB
     * runtime with 12 cpus will have buffer size of 2MB
     *
     * @return transaction log writer buffer size.
     */
    private static int calculateLogBufferSize()
    {
        return (int) ByteUnit.kibiBytes( min( (getRuntime().availableProcessors() / 4) + 1, 8 ) * 512 );
    }

    private static class UncloseableChannel extends DelegatingStoreChannel<LogVersionedStoreChannel> implements LogVersionedStoreChannel
    {
        UncloseableChannel( LogVersionedStoreChannel channel )
        {
            super( channel );
        }

        @Override
        public long getVersion()
        {
            return delegate.getVersion();
        }

        @Override
        public byte getLogFormatVersion()
        {
            return delegate.getLogFormatVersion();
        }

        @Override
        public void close() throws IOException
        {
            // do not close since channel is shared
        }
    }
}
