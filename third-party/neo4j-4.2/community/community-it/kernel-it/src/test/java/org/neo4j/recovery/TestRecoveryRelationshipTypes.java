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
package org.neo4j.recovery;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Iterator;

import org.neo4j.dbms.api.DatabaseManagementService;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.RelationshipType;
import org.neo4j.graphdb.Transaction;
import org.neo4j.kernel.impl.MyRelTypes;
import org.neo4j.kernel.impl.transaction.log.checkpoint.CheckPointer;
import org.neo4j.kernel.impl.transaction.log.checkpoint.SimpleTriggerInfo;
import org.neo4j.kernel.internal.GraphDatabaseAPI;
import org.neo4j.test.TestDatabaseManagementServiceBuilder;
import org.neo4j.test.extension.Inject;
import org.neo4j.test.extension.testdirectory.TestDirectoryExtension;
import org.neo4j.test.rule.TestDirectory;

import static java.lang.Runtime.getRuntime;
import static java.lang.System.exit;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.neo4j.configuration.GraphDatabaseSettings.DEFAULT_DATABASE_NAME;
import static org.neo4j.test.proc.ProcessUtil.getClassPath;
import static org.neo4j.test.proc.ProcessUtil.getJavaExecutable;

@TestDirectoryExtension
class TestRecoveryRelationshipTypes
{
    @Inject
    private TestDirectory testDirectory;

    @Test
    void recoverNeoAndHavingAllRelationshipTypesAfterRecovery() throws Exception
    {
        // Given (create transactions and kill process, leaving it needing for recovery)
        Path storeDir = testDirectory.homePath();
        assertEquals( 0, getRuntime().exec( new String[]{
                getJavaExecutable().toString(), "-Djava.awt.headless=true", "-cp",
                getClassPath(), getClass().getName(), storeDir.toAbsolutePath().toString()} ).waitFor() );

        // When
        DatabaseManagementService managementService = new TestDatabaseManagementServiceBuilder( storeDir ).build();
        GraphDatabaseService db = managementService.database( DEFAULT_DATABASE_NAME );

        // Then
        try ( Transaction transaction = db.beginTx() )
        {
            Iterator<RelationshipType> typeResourceIterator = transaction.getAllRelationshipTypes().iterator();
            assertEquals( MyRelTypes.TEST.name(), typeResourceIterator.next().name() );
        }
        finally
        {
            managementService.shutdown();
        }
    }

    public static void main( String[] args ) throws IOException
    {
        if ( args.length != 1 )
        {
            exit( 1 );
        }

        Path storeDir = Path.of( args[0] ).toAbsolutePath();
        DatabaseManagementService managementService = new TestDatabaseManagementServiceBuilder( storeDir ).build();
        GraphDatabaseService db = managementService.database( DEFAULT_DATABASE_NAME );
        try ( Transaction tx = db.beginTx() )
        {
            tx.createNode().createRelationshipTo( tx.createNode(), MyRelTypes.TEST );
            tx.commit();
        }

        CheckPointer checkPointer = ((GraphDatabaseAPI) db).getDependencyResolver().resolveDependency( CheckPointer.class );
        checkPointer.forceCheckPoint( new SimpleTriggerInfo( "test" ) );

        exit( 0 );
    }
}
