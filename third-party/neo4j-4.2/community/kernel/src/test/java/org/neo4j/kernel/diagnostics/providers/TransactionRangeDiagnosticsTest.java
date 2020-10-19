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
package org.neo4j.kernel.diagnostics.providers;

import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;

import org.neo4j.collection.Dependencies;
import org.neo4j.kernel.database.Database;
import org.neo4j.kernel.impl.transaction.log.entry.LogEntryVersion;
import org.neo4j.kernel.impl.transaction.log.entry.LogHeader;
import org.neo4j.kernel.impl.transaction.log.files.LogFiles;
import org.neo4j.kernel.impl.transaction.log.files.TransactionLogFiles;
import org.neo4j.logging.AssertableLogProvider;
import org.neo4j.logging.Log;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.neo4j.kernel.impl.transaction.log.entry.LogVersions.CURRENT_FORMAT_LOG_HEADER_SIZE;
import static org.neo4j.logging.LogAssertions.assertThat;

class TransactionRangeDiagnosticsTest
{
    @Test
    void shouldLogCorrectTransactionLogDiagnosticsForNoTransactionLogs()
    {
        // GIVEN
        Database database = databaseWithLogFilesContainingLowestTxId( noLogs() );
        AssertableLogProvider logProvider = new AssertableLogProvider();
        Log logger = logProvider.getLog( getClass() );

        // WHEN
        new TransactionRangeDiagnostics( database ).dump( logger::info );

        // THEN
        assertThat( logProvider )
                .containsMessages( "Transaction log files stored on file store:" )
                .containsMessages( "No transactions" );
    }

    @Test
    void shouldLogCorrectTransactionLogDiagnosticsForTransactionsInOldestLog() throws Exception
    {
        // GIVEN
        long logVersion = 2;
        long prevLogLastTxId = 45;
        Database database = databaseWithLogFilesContainingLowestTxId(
                logWithTransactions( logVersion, prevLogLastTxId ) );
        AssertableLogProvider logProvider = new AssertableLogProvider();
        Log logger = logProvider.getLog( getClass() );

        // WHEN
        new TransactionRangeDiagnostics( database ).dump( logger::info );

        // THEN
        assertThat( logProvider ).containsMessages( "transaction " + (prevLogLastTxId + 1),
                                                    "version " + logVersion );
    }

    @Test
    void shouldLogCorrectTransactionLogDiagnosticsForTransactionsInSecondOldestLog() throws Exception
    {
        // GIVEN
        long logVersion = 2;
        long prevLogLastTxId = 45;
        Database database = databaseWithLogFilesContainingLowestTxId(
                logWithTransactionsInNextToOldestLog( logVersion, prevLogLastTxId ) );
        AssertableLogProvider logProvider = new AssertableLogProvider();
        Log logger = logProvider.getLog( getClass() );

        // WHEN
        new TransactionRangeDiagnostics( database ).dump( logger::info );

        // THEN
        assertThat( logProvider ).containsMessages( "transaction " + (prevLogLastTxId + 1), "version " + (logVersion + 1) );
    }

    private static Database databaseWithLogFilesContainingLowestTxId( LogFiles files )
    {
        Dependencies dependencies = mock( Dependencies.class );
        when( dependencies.resolveDependency( LogFiles.class ) ).thenReturn( files );
        Database database = mock( Database.class );
        when( database.getDependencyResolver() ).thenReturn( dependencies );
        return database;
    }

    private static LogFiles logWithTransactionsInNextToOldestLog( long logVersion, long prevLogLastTxId )
            throws IOException
    {
        LogFiles files = logWithTransactions( logVersion + 1, prevLogLastTxId );
        when( files.getLowestLogVersion() ).thenReturn( logVersion );
        when( files.hasAnyEntries( logVersion ) ).thenReturn( false );
        when( files.versionExists( logVersion ) ).thenReturn( true );
        return files;
    }

    private static LogFiles logWithTransactions( long logVersion, long headerTxId ) throws IOException
    {
        LogFiles files = mock( TransactionLogFiles.class );
        when( files.logFilesDirectory() ).thenReturn( new File( "." ) );
        when( files.getLowestLogVersion() ).thenReturn( logVersion );
        when( files.hasAnyEntries( logVersion ) ).thenReturn( true );
        when( files.versionExists( logVersion ) ).thenReturn( true );
        when( files.extractHeader( logVersion ) )
                .thenReturn( new LogHeader( LogEntryVersion.LATEST.version(), logVersion, headerTxId, CURRENT_FORMAT_LOG_HEADER_SIZE ) );
        return files;
    }

    private static LogFiles noLogs()
    {
        LogFiles files = mock( TransactionLogFiles.class );
        when( files.getLowestLogVersion() ).thenReturn( -1L );
        when( files.logFilesDirectory() ).thenReturn( new File( "." ) );
        return files;
    }
}
