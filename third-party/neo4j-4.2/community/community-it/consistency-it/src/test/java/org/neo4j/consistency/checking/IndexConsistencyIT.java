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
package org.neo4j.consistency.checking;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.io.FileFilter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.neo4j.configuration.Config;
import org.neo4j.consistency.ConsistencyCheckService;
import org.neo4j.consistency.checking.full.ConsistencyCheckIncompleteException;
import org.neo4j.consistency.checking.full.ConsistencyFlags;
import org.neo4j.dbms.api.DatabaseManagementService;
import org.neo4j.graphdb.Label;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Transaction;
import org.neo4j.internal.helpers.collection.Pair;
import org.neo4j.io.fs.DefaultFileSystemAbstraction;
import org.neo4j.io.fs.FileSystemAbstraction;
import org.neo4j.io.layout.DatabaseLayout;
import org.neo4j.kernel.impl.transaction.log.checkpoint.CheckPointer;
import org.neo4j.kernel.impl.transaction.log.checkpoint.SimpleTriggerInfo;
import org.neo4j.kernel.impl.transaction.state.DefaultIndexProviderMap;
import org.neo4j.kernel.internal.GraphDatabaseAPI;
import org.neo4j.logging.AssertableLogProvider;
import org.neo4j.test.extension.DbmsExtension;
import org.neo4j.test.extension.Inject;
import org.neo4j.test.extension.RandomExtension;
import org.neo4j.test.rule.RandomRule;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.neo4j.configuration.GraphDatabaseSettings.logs_directory;
import static org.neo4j.internal.helpers.progress.ProgressMonitorFactory.NONE;
import static org.neo4j.io.fs.FileUtils.copyRecursively;
import static org.neo4j.test.TestLabels.LABEL_ONE;
import static org.neo4j.test.TestLabels.LABEL_THREE;
import static org.neo4j.test.TestLabels.LABEL_TWO;

@DbmsExtension
@ExtendWith( RandomExtension.class )
class IndexConsistencyIT
{
    @Inject
    private GraphDatabaseAPI db;
    @Inject
    private DatabaseManagementService managementService;
    @Inject
    private CheckPointer checkPointer;
    @Inject
    private DefaultIndexProviderMap indexProviderMap;

    @Inject
    private RandomRule random;

    private final AssertableLogProvider log = new AssertableLogProvider();
    private static final Label[] LABELS = new Label[]{LABEL_ONE, LABEL_TWO, LABEL_THREE};
    private static final String PROPERTY_KEY = "numericProperty";
    private static final double DELETE_RATIO = 0.2;
    private static final double UPDATE_RATIO = 0.2;
    private static final int NODE_COUNT_BASELINE = 10;
    private final FileFilter SOURCE_COPY_FILE_FILTER = file -> file.isDirectory() || file.getName().startsWith( "index" );

    @Test
    void reportNotCleanNativeIndex() throws IOException, ConsistencyCheckIncompleteException
    {
        DatabaseLayout databaseLayout = db.databaseLayout();
        someData();
        checkPointer.forceCheckPoint( new SimpleTriggerInfo( "forcedCheckpoint" ) );
        Path indexesCopy = databaseLayout.file( "indexesCopy" );
        Path indexSources = indexProviderMap.getDefaultProvider().directoryStructure().rootDirectory();
        copyRecursively( indexSources.toFile(), indexesCopy.toFile(), SOURCE_COPY_FILE_FILTER );

        try ( Transaction tx = db.beginTx() )
        {
            createNewNode( tx, new Label[]{LABEL_ONE} );
            tx.commit();
        }

        managementService.shutdown();

        copyRecursively( indexesCopy.toFile(), indexSources.toFile() );

        ConsistencyCheckService.Result result = fullConsistencyCheck();
        assertFalse( result.isSuccessful(), "Expected consistency check to fail" );
        assertThat( readReport( result ) ).contains(
                "WARN  Index was dirty on startup which means it was not shutdown correctly and need to be cleaned up with a successful recovery." );
    }

    @Test
    void reportNotCleanNativeIndexWithCorrectData() throws IOException, ConsistencyCheckIncompleteException
    {
        DatabaseLayout databaseLayout = db.databaseLayout();
        someData();
        checkPointer.forceCheckPoint( new SimpleTriggerInfo( "forcedCheckpoint" ) );
        Path indexesCopy = databaseLayout.file( "indexesCopy" );
        Path indexSources = indexProviderMap.getDefaultProvider().directoryStructure().rootDirectory();
        copyRecursively( indexSources.toFile(), indexesCopy.toFile(), SOURCE_COPY_FILE_FILTER );

        managementService.shutdown();

        copyRecursively( indexesCopy.toFile(), indexSources.toFile() );

        ConsistencyCheckService.Result result = fullConsistencyCheck();
        assertTrue( result.isSuccessful(), "Expected consistency check to fail" );
        assertThat( readReport( result ) ).contains(
                "WARN  Index was dirty on startup which means it was not shutdown correctly and need to be cleaned up with a successful recovery." );
    }

    private String readReport( ConsistencyCheckService.Result result ) throws IOException
    {
        return Files.readString( result.reportFile() );
    }

    void someData()
    {
        someData( 50 );
    }

    void someData( int numberOfModifications )
    {
        List<Pair<Long,Label[]>> existingNodes;
        existingNodes = new ArrayList<>();
        try ( Transaction tx = db.beginTx() )
        {
            randomModifications( tx, existingNodes, numberOfModifications );
            tx.commit();
        }
        try ( Transaction tx = db.beginTx() )
        {
            tx.schema().indexFor( LABEL_ONE ).on( PROPERTY_KEY ).create();
            tx.commit();
        }
        try ( Transaction tx = db.beginTx() )
        {
            tx.schema().awaitIndexesOnline( 1, TimeUnit.MINUTES );
            tx.commit();
        }
    }

    private void randomModifications( Transaction tx, List<Pair<Long,Label[]>> existingNodes,
            int numberOfModifications )
    {
        for ( int i = 0; i < numberOfModifications; i++ )
        {
            double selectModification = random.nextDouble();
            if ( existingNodes.size() < NODE_COUNT_BASELINE || selectModification >= DELETE_RATIO + UPDATE_RATIO )
            {
                createNewNode( tx, existingNodes );
            }
            else if ( selectModification < DELETE_RATIO )
            {
                deleteExistingNode( tx, existingNodes );
            }
            else
            {
                modifyLabelsOnExistingNode( tx, existingNodes );
            }
        }
    }

    private void createNewNode( Transaction transaction, List<Pair<Long,Label[]>> existingNodes )
    {
        Label[] labels = randomLabels();
        Node node = createNewNode( transaction, labels );
        existingNodes.add( Pair.of( node.getId(), labels ) );
    }

    private Node createNewNode( Transaction tx, Label[] labels )
    {
        Node node = tx.createNode( labels );
        node.setProperty( PROPERTY_KEY, random.nextInt() );
        return node;
    }

    private void modifyLabelsOnExistingNode( Transaction transaction, List<Pair<Long,Label[]>> existingNodes )
    {
        int targetIndex = random.nextInt( existingNodes.size() );
        Pair<Long,Label[]> existingPair = existingNodes.get( targetIndex );
        long nodeId = existingPair.first();
        Node node = transaction.getNodeById( nodeId );
        node.getLabels().forEach( node::removeLabel );
        Label[] newLabels = randomLabels();
        for ( Label label : newLabels )
        {
            node.addLabel( label );
        }
        existingNodes.remove( targetIndex );
        existingNodes.add( Pair.of( nodeId, newLabels ) );
    }

    private void deleteExistingNode( Transaction transaction, List<Pair<Long,Label[]>> existingNodes )
    {
        int targetIndex = random.nextInt( existingNodes.size() );
        Pair<Long,Label[]> existingPair = existingNodes.get( targetIndex );
        Node node = transaction.getNodeById( existingPair.first() );
        node.delete();
        existingNodes.remove( targetIndex );
    }

    private Label[] randomLabels()
    {
        List<Label> labels = new ArrayList<>( LABELS.length );
        for ( Label label : LABELS )
        {
            if ( random.nextBoolean() )
            {
                labels.add( label );
            }
        }
        return labels.toArray( new Label[0] );
    }

    private ConsistencyCheckService.Result fullConsistencyCheck()
            throws ConsistencyCheckIncompleteException, IOException
    {
        try ( FileSystemAbstraction fsa = new DefaultFileSystemAbstraction() )
        {
            ConsistencyCheckService service = new ConsistencyCheckService();
            DatabaseLayout databaseLayout = db.databaseLayout();
            Config config = Config.defaults( logs_directory, databaseLayout.databaseDirectory() );
            return service.runFullConsistencyCheck( databaseLayout, config, NONE, log, fsa, false, ConsistencyFlags.DEFAULT );
        }
    }
}
