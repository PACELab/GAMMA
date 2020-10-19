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

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.extension.ConditionEvaluationResult;
import org.junit.jupiter.api.extension.ExecutionCondition;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.api.parallel.ResourceLock;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.File;
import java.util.LinkedList;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import org.neo4j.internal.helpers.collection.Pair;
import org.neo4j.test.extension.testdirectory.TestDirectoryExtension;
import org.neo4j.test.rule.TestDirectory;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.extension.ConditionEvaluationResult.disabled;
import static org.neo4j.test.extension.ExecutionSharedContext.CONTEXT;
import static org.neo4j.test.extension.ExecutionSharedContext.CREATED_TEST_FILE_PAIRS_KEY;
import static org.neo4j.test.extension.ExecutionSharedContext.LOCKED_TEST_FILE_KEY;
import static org.neo4j.test.extension.ExecutionSharedContext.SHARED_RESOURCE;
import static org.neo4j.test.extension.ExecutionSharedContext.SUCCESSFUL_TEST_FILE_KEY;

/**
 * This test is disabled by default and not executed directly by test runner.
 * It will be executed by a specific test executor as part of extensions lifecycle testing.
 * @see TestDirectoryExtensionTestSupport#failedTestShouldKeepDirectory()
 */
@TestDirectoryExtension
@ExtendWith( DirectoryExtensionLifecycleVerificationTest.ConfigurationParameterCondition.class )
@ResourceLock( SHARED_RESOURCE )
class DirectoryExtensionLifecycleVerificationTest
{
    @Inject
    private TestDirectory directory;

    @Test
    void executeAndCleanupDirectory()
    {
        File file = directory.createFile( "a" );
        assertTrue( file.exists() );
        CONTEXT.setValue( SUCCESSFUL_TEST_FILE_KEY, file );
    }

    @Test
    void failAndKeepDirectory()
    {
        File file = directory.createFile( "b" );
        CONTEXT.setValue( CREATED_TEST_FILE_PAIRS_KEY, file );
        throw new RuntimeException( "simulate test failure" );
    }

    @Test
    void lockFileAndFailToDeleteDirectory()
    {
        File nonDeletableDirectory = directory.directory( "c" );
        CONTEXT.setValue( LOCKED_TEST_FILE_KEY, nonDeletableDirectory );
        assertTrue( nonDeletableDirectory.setReadable( false, false ) );
    }

    @Nested
    @TestInstance( TestInstance.Lifecycle.PER_CLASS )
    class PerClassTest extends SecondTestFailTest
    {
    }

    @Nested
    @TestInstance( TestInstance.Lifecycle.PER_METHOD )
    class PerMethodTest extends SecondTestFailTest
    {
    }

    static class SecondTestFailTest
    {
        @Inject
        TestDirectory testDirectory;

        @Test
        void createAFileAndThenPass()
        {
            createFileSaveAndFailIfNeeded( Boolean.FALSE );
        }

        @Test
        void createAFileAndThenFail()
        {
            createFileSaveAndFailIfNeeded( Boolean.TRUE );
        }

        @Test
        void createAnotherFileAndThenPass()
        {
            createFileSaveAndFailIfNeeded( Boolean.FALSE );
        }

        @ValueSource( booleans = {false, true, false} )
        @ParameterizedTest
        void createFileSaveAndFailIfNeeded( Boolean fail )
        {
            var filename = UUID.randomUUID().toString();
            var file = testDirectory.createFile( filename );
            List<Pair<File,Boolean>> pairs = CONTEXT.getValue( CREATED_TEST_FILE_PAIRS_KEY );
            pairs = pairs == null ? new LinkedList<>() : pairs;
            pairs.add( Pair.of( file, fail ) );
            CONTEXT.setValue( CREATED_TEST_FILE_PAIRS_KEY, pairs );
            if ( fail )
            {
                fail();
            }
        }
    }

    static class ConfigurationParameterCondition implements ExecutionCondition
    {
        static final String TEST_TOGGLE = "testToggle";

        @Override
        public ConditionEvaluationResult evaluateExecutionCondition( ExtensionContext context )
        {
            Optional<String> option = context.getConfigurationParameter( TEST_TOGGLE );
            return option.map( ConditionEvaluationResult::enabled ).orElseGet( () -> disabled( "configuration parameter not present" ) );
        }
    }
}
