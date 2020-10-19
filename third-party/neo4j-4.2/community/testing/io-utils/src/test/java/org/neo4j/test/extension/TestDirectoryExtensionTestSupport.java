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
package org.neo4j.test.extension;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.junit.jupiter.api.parallel.ResourceLock;
import org.junit.platform.engine.TestExecutionResult;
import org.junit.platform.launcher.Launcher;
import org.junit.platform.launcher.LauncherDiscoveryRequest;
import org.junit.platform.launcher.TestExecutionListener;
import org.junit.platform.launcher.TestIdentifier;
import org.junit.platform.launcher.core.LauncherDiscoveryRequestBuilder;
import org.junit.platform.launcher.core.LauncherFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import org.neo4j.internal.helpers.collection.Pair;
import org.neo4j.io.fs.DefaultFileSystemAbstraction;
import org.neo4j.io.fs.FileUtils;
import org.neo4j.test.extension.testdirectory.TestDirectoryExtension;
import org.neo4j.test.rule.TestDirectory;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.platform.engine.TestExecutionResult.Status.FAILED;
import static org.junit.platform.engine.discovery.DiscoverySelectors.selectClass;
import static org.junit.platform.engine.discovery.DiscoverySelectors.selectMethod;
import static org.neo4j.test.extension.DirectoryExtensionLifecycleVerificationTest.ConfigurationParameterCondition.TEST_TOGGLE;
import static org.neo4j.test.extension.ExecutionSharedContext.CONTEXT;
import static org.neo4j.test.extension.ExecutionSharedContext.CREATED_TEST_FILE_PAIRS_KEY;
import static org.neo4j.test.extension.ExecutionSharedContext.LOCKED_TEST_FILE_KEY;
import static org.neo4j.test.extension.ExecutionSharedContext.SUCCESSFUL_TEST_FILE_KEY;

@TestDirectoryExtension
@ResourceLock( ExecutionSharedContext.SHARED_RESOURCE )
class TestDirectoryExtensionTestSupport
{
    @Inject
    TestDirectory testDirectory;
    @Inject
    DefaultFileSystemAbstraction fileSystem;

    @Test
    void testDirectoryInjectionWorks()
    {
        assertNotNull( testDirectory );
    }

    @Test
    void testDirectoryInitialisedForUsage()
    {
        File directory = testDirectory.homeDir();
        assertNotNull( directory );
        assertTrue( directory.exists() );
        Path targetTestData = Paths.get( "target", "test data" );
        assertTrue( directory.getAbsolutePath().contains( targetTestData.toString() ) );
    }

    @Test
    void testDirectoryUsesFileSystemFromExtension()
    {
        assertSame( fileSystem, testDirectory.getFileSystem() );
    }

    @Test
    void createTestFile()
    {
        File file = testDirectory.createFile( "a" );
        assertEquals( "a", file.getName() );
        assertTrue( fileSystem.fileExists( file ) );
    }

    @Test
    void failedTestShouldKeepDirectory()
    {
        CONTEXT.clear();
        execute( "failAndKeepDirectory" );
        File failedFile = CONTEXT.getValue( CREATED_TEST_FILE_PAIRS_KEY );
        assertNotNull( failedFile );
        assertTrue( failedFile.exists() );
    }

    @Test
    void successfulTestShouldCleanupDirectory()
    {
        CONTEXT.clear();
        execute( "executeAndCleanupDirectory" );
        File greenTestFail = CONTEXT.getValue( SUCCESSFUL_TEST_FILE_KEY );
        assertNotNull( greenTestFail );
        assertFalse( greenTestFail.exists() );
    }

    @Test
    @EnabledOnOs( OS.LINUX )
    @DisabledForRoot
    void exceptionOnDirectoryDeletionIncludeTestDisplayName() throws IOException
    {
        CONTEXT.clear();
        FailedTestExecutionListener failedTestListener = new FailedTestExecutionListener();
        execute( "lockFileAndFailToDeleteDirectory", failedTestListener );
        File lockedFile = CONTEXT.getValue( LOCKED_TEST_FILE_KEY );

        assertNotNull( lockedFile );
        assertTrue( lockedFile.setReadable( true, true ) );
        FileUtils.deleteRecursively( lockedFile );
        failedTestListener.assertTestObserver();
    }

    @Test
    void failedTestShouldKeepDirectoryInPerClassLifecycle()
    {
        List<Pair<File,Boolean>> pairs = executeAndReturnCreatedFiles( DirectoryExtensionLifecycleVerificationTest.PerClassTest.class, 6 );
        for ( var pair : pairs )
        {
            assertThat( pair.first() ).exists();
        }
    }

    @Test
    void failedTestShouldNotKeepDirectoryInPerMethodLifecycle()
    {
        List<Pair<File,Boolean>> pairs = executeAndReturnCreatedFiles( DirectoryExtensionLifecycleVerificationTest.PerMethodTest.class, 6 );
        for ( var pair : pairs )
        {
            if ( pair.other() )
            {
                assertThat( pair.first() ).exists();
            }
            else
            {
                assertThat( pair.first() ).doesNotExist();
            }
        }
    }

    private static List<Pair<File,Boolean>> executeAndReturnCreatedFiles( Class testClass, int count )
    {
        CONTEXT.clear();
        executeClass( testClass );
        List<Pair<File,Boolean>> pairs = CONTEXT.getValue( CREATED_TEST_FILE_PAIRS_KEY );
        assertNotNull( pairs );
        assertThat( pairs.size() ).isEqualTo( count );
        return pairs;
    }

    private static void execute( String testName, TestExecutionListener... testExecutionListeners )
    {
        LauncherDiscoveryRequest discoveryRequest = LauncherDiscoveryRequestBuilder.request()
                .selectors( selectMethod( DirectoryExtensionLifecycleVerificationTest.class, testName ))
                .configurationParameter( TEST_TOGGLE, "true" )
                .build();
        execute( discoveryRequest, testExecutionListeners );
    }

    private static void executeClass( Class testClass, TestExecutionListener... testExecutionListeners )
    {
        LauncherDiscoveryRequest discoveryRequest =
                LauncherDiscoveryRequestBuilder.request().selectors( selectClass( testClass ) ).configurationParameter( TEST_TOGGLE, "true" ).build();
        execute( discoveryRequest, testExecutionListeners );
    }

    private static void execute( LauncherDiscoveryRequest discoveryRequest, TestExecutionListener... testExecutionListeners )
    {
        Launcher launcher = LauncherFactory.create();
        launcher.execute( discoveryRequest, testExecutionListeners );
    }

    private static class FailedTestExecutionListener implements TestExecutionListener
    {
        private int resultsObserved;

        @Override
        public void executionFinished( TestIdentifier testIdentifier, TestExecutionResult testExecutionResult )
        {
            if ( testExecutionResult.getStatus() == FAILED )
            {
                resultsObserved++;
                String exceptionMessage = testExecutionResult.getThrowable().map( Throwable::getMessage ).orElse( "" );
                assertThat( exceptionMessage ).contains( "Fail to cleanup test directory for lockFileAndFailToDeleteDirectory" );
            }
        }

        void assertTestObserver()
        {
            assertEquals( 1, resultsObserved );
        }
    }
}
