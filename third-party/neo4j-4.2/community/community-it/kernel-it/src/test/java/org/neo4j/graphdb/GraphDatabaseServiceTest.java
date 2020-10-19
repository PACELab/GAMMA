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
package org.neo4j.graphdb;

import org.junit.After;
import org.junit.Before;
import org.junit.ClassRule;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.rules.RuleChain;

import java.time.Duration;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import org.neo4j.configuration.GraphDatabaseSettings;
import org.neo4j.dbms.api.DatabaseManagementService;
import org.neo4j.kernel.DeadlockDetectedException;
import org.neo4j.kernel.availability.DatabaseAvailability;
import org.neo4j.kernel.impl.MyRelTypes;
import org.neo4j.test.Barrier;
import org.neo4j.test.TestDatabaseManagementServiceBuilder;
import org.neo4j.test.rule.DbmsRule;
import org.neo4j.test.rule.ImpermanentDbmsRule;
import org.neo4j.test.rule.OtherThreadRule;
import org.neo4j.test.rule.TestDirectory;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;
import static org.neo4j.configuration.GraphDatabaseSettings.DEFAULT_DATABASE_NAME;

public class GraphDatabaseServiceTest
{
    @ClassRule
    public static final DbmsRule globalDb = new ImpermanentDbmsRule()
                                            .withSetting( GraphDatabaseSettings.shutdown_transaction_end_timeout, Duration.ofSeconds( 10 ) );

    private final ExpectedException exception = ExpectedException.none();
    private final TestDirectory testDirectory = TestDirectory.testDirectory();
    private final OtherThreadRule t2 = new OtherThreadRule();
    private final OtherThreadRule t3 = new OtherThreadRule();

    @Rule
    public RuleChain chain = RuleChain.outerRule( testDirectory ).around( exception );
    private DatabaseManagementService managementService;

    @Before
    public void before()
    {
        t2.init( "T2-" + getClass().getName() );
        t3.init( "T3-" + getClass().getName() );
    }

    @After
    public void after()
    {
        t2.close();
        t3.close();
    }

    @Test
    public void givenShutdownDatabaseWhenBeginTxThenExceptionIsThrown()
    {
        // Given
        GraphDatabaseService db = getTemporaryDatabase();

        managementService.shutdown();

        // Expect
        exception.expect( DatabaseShutdownException.class );

        // When
        db.beginTx();
    }

    @Test
    public void givenDatabaseAndStartedTxWhenShutdownThenWaitForTxToFinish() throws Exception
    {
        // Given
        final GraphDatabaseService db = getTemporaryDatabase();

        // When
        Barrier.Control barrier = new Barrier.Control();
        Future<Object> txFuture = t2.execute( () ->
        {
            try ( Transaction tx = db.beginTx() )
            {
                barrier.reached();
                tx.createNode();
                tx.commit();
            }
            return null;
        } );

        // i.e. wait for transaction to start
        barrier.await();

        // now there's a transaction open, blocked on continueTxSignal
        Future<Object> shutdownFuture = t3.execute( () ->
        {
            managementService.shutdown();
            return null;
        } );
        t3.get().waitUntilWaiting( location -> location.isAt( DatabaseAvailability.class, "stop" ) );
        barrier.release();
        try
        {
            txFuture.get();
        }
        catch ( ExecutionException e )
        {
            // expected
        }
        shutdownFuture.get();
    }

    @Test
    public void terminateTransactionThrowsExceptionOnNextOperation()
    {
        // Given
        final GraphDatabaseService db = globalDb;

        try ( Transaction tx = db.beginTx() )
        {
            tx.terminate();
            assertThrows( TransactionTerminatedException.class, tx::createNode );
        }
    }

    @Test
    public void givenDatabaseAndStartedTxWhenShutdownAndStartNewTxThenBeginTxTimesOut() throws Exception
    {
        // Given
        GraphDatabaseService db = getTemporaryDatabase();

        // When
        Barrier.Control barrier = new Barrier.Control();
        t2.execute( () ->
        {
            try ( Transaction tx = db.beginTx() )
            {
                barrier.reached(); // <-- this triggers t3 to start a managementService.shutdown()
            }
            return null;
        } );

        barrier.await();
        Future<Object> shutdownFuture = t3.execute( () ->
        {
            managementService.shutdown();
            return null;
        } );
        t3.get().waitUntilWaiting( location -> location.isAt( DatabaseAvailability.class, "stop" ) );
        barrier.release(); // <-- this triggers t2 to continue its transaction
        shutdownFuture.get();

        assertThrows( DatabaseShutdownException.class, db::beginTx );
    }

    @Test
    public void shouldLetDetectedDeadlocksDuringCommitBeThrownInTheirOriginalForm() throws Exception
    {
        // GIVEN a database with a couple of entities:
        // (n1) --> (r1) --> (r2) --> (r3)
        // (n2)
        GraphDatabaseService db = globalDb;
        Node n1 = createNode( db );
        Node n2 = createNode( db );
        Relationship r3 = createRelationship( db, n1 );
        Relationship r2 = createRelationship( db, n1 );
        Relationship r1 = createRelationship( db, n1 );

        // WHEN creating a deadlock scenario where the final deadlock would have happened due to locks
        //      acquired during linkage of relationship records
        //
        // (r1) <-- (t1)
        //   |       ^
        //   v       |
        // (t2) --> (n2)
        Transaction t1Tx = db.beginTx();
        Transaction t2Tx = t2.execute( beginTx( db ) ).get();
        // (t1) <-- (n2)
        t1Tx.getNodeById( n2.getId() ).setProperty( "locked", "indeed" );
        // (t2) <-- (r1)
        t2.execute( setProperty( t2Tx.getRelationshipById( r1.getId() ), "locked", "absolutely" ) ).get();
        // (t2) --> (n2)
        Future<Void> t2n2Wait = t2.execute( setProperty( t2Tx.getNodeById( n2.getId() ), "locked", "In my dreams" ) );
        t2.get().waitUntilWaiting();
        // (t1) --> (r1) although delayed until commit, this is accomplished by deleting an adjacent
        //               relationship so that its surrounding relationships are locked at commit time.
        t1Tx.getRelationshipById( r2.getId() ).delete();
        try
        {
            t1Tx.commit();
            fail( "Should throw exception about deadlock" );
        }
        catch ( Exception e )
        {
            assertEquals( DeadlockDetectedException.class, e.getClass() );
        }
        finally
        {
            t2n2Wait.get();
            t2.execute( close( t2Tx ) ).get();
        }
    }

    /**
     * GitHub issue #5996
     */
    @Test
    public void terminationOfClosedTransactionDoesNotInfluenceNextTransaction()
    {
        GraphDatabaseService db = globalDb;

        try ( Transaction tx = db.beginTx() )
        {
            tx.createNode();
            tx.commit();
        }

        Transaction transaction = db.beginTx();
        try ( Transaction tx = transaction )
        {
            tx.createNode();
            tx.commit();
        }
        transaction.terminate();

        try ( Transaction tx = db.beginTx() )
        {
            assertThat( tx.getAllNodes() ).hasSize( 2 );
            tx.commit();
        }
    }

    private Callable<Transaction> beginTx( final GraphDatabaseService db )
    {
        return db::beginTx;
    }

    private Callable<Void> setProperty( final Entity entity, final String key, final String value )
    {
        return () ->
        {
            entity.setProperty( key, value );
            return null;
        };
    }

    private Callable<Void> close( final Transaction tx )
    {
        return () ->
        {
            tx.close();
            return null;
        };
    }

    private Relationship createRelationship( GraphDatabaseService db, Node node )
    {
        try ( Transaction tx = db.beginTx() )
        {
            Relationship rel = tx.getNodeById( node.getId() ).createRelationshipTo( node, MyRelTypes.TEST );
            tx.commit();
            return rel;
        }
    }

    private Node createNode( GraphDatabaseService db )
    {
        try ( Transaction tx = db.beginTx() )
        {
            Node node = tx.createNode();
            tx.commit();
            return node;
        }
    }

    private GraphDatabaseService getTemporaryDatabase()
    {
        managementService = new TestDatabaseManagementServiceBuilder( testDirectory.directoryPath( "impermanent" ) ).impermanent()
                .setConfig( GraphDatabaseSettings.shutdown_transaction_end_timeout, Duration.ofSeconds( 10 ) ).build();
        return managementService.database( DEFAULT_DATABASE_NAME );
    }
}
