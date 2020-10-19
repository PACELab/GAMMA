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
package org.neo4j.upgrade;

import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.RuleChain;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Collections;

import org.neo4j.collection.Dependencies;
import org.neo4j.common.ProgressReporter;
import org.neo4j.configuration.Config;
import org.neo4j.configuration.GraphDatabaseSettings;
import org.neo4j.consistency.checking.full.ConsistencyCheckIncompleteException;
import org.neo4j.dbms.api.DatabaseManagementService;
import org.neo4j.exceptions.KernelException;
import org.neo4j.internal.batchimport.BatchImporterFactory;
import org.neo4j.internal.recordstorage.RecordStorageEngineFactory;
import org.neo4j.io.fs.FileSystemAbstraction;
import org.neo4j.io.layout.DatabaseLayout;
import org.neo4j.io.layout.Neo4jLayout;
import org.neo4j.io.pagecache.PageCache;
import org.neo4j.io.pagecache.tracing.DefaultPageCacheTracer;
import org.neo4j.kernel.api.index.IndexProvider;
import org.neo4j.kernel.impl.store.format.RecordFormatSelector;
import org.neo4j.kernel.impl.store.format.RecordFormats;
import org.neo4j.kernel.impl.store.format.standard.StandardV3_4;
import org.neo4j.kernel.impl.storemigration.IdGeneratorMigrator;
import org.neo4j.kernel.impl.storemigration.LegacyTransactionLogsLocator;
import org.neo4j.kernel.impl.storemigration.LogsUpgrader;
import org.neo4j.kernel.impl.storemigration.MigrationTestUtils;
import org.neo4j.kernel.impl.storemigration.RecordStorageMigrator;
import org.neo4j.kernel.impl.storemigration.RecordStoreVersionCheck;
import org.neo4j.kernel.impl.storemigration.StoreUpgrader;
import org.neo4j.logging.NullLogProvider;
import org.neo4j.logging.internal.LogService;
import org.neo4j.logging.internal.NullLogService;
import org.neo4j.monitoring.Monitors;
import org.neo4j.scheduler.JobScheduler;
import org.neo4j.storageengine.api.StoreVersionCheck;
import org.neo4j.storageengine.migration.MigrationProgressMonitor;
import org.neo4j.storageengine.migration.SchemaIndexMigrator;
import org.neo4j.storageengine.migration.StoreMigrationParticipant;
import org.neo4j.test.TestDatabaseManagementServiceBuilder;
import org.neo4j.test.rule.PageCacheRule;
import org.neo4j.test.rule.TestDirectory;
import org.neo4j.test.rule.fs.DefaultFileSystemRule;
import org.neo4j.test.scheduler.ThreadPoolJobScheduler;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static org.neo4j.configuration.GraphDatabaseSettings.DEFAULT_DATABASE_NAME;
import static org.neo4j.configuration.GraphDatabaseSettings.allow_upgrade;
import static org.neo4j.consistency.store.StoreAssertions.assertConsistentStore;
import static org.neo4j.io.pagecache.tracing.PageCacheTracer.NULL;
import static org.neo4j.kernel.impl.storemigration.MigrationTestUtils.checkNeoStoreHasFormatVersion;
import static org.neo4j.memory.EmptyMemoryTracker.INSTANCE;
import static org.neo4j.storageengine.api.StorageEngineFactory.selectStorageEngine;

@RunWith( Parameterized.class )
public class StoreUpgraderInterruptionTestIT
{
    private static final Config CONFIG = Config.defaults( GraphDatabaseSettings.pagecache_memory, "8m" );

    private final TestDirectory directory = TestDirectory.testDirectory();
    private final DefaultFileSystemRule fileSystemRule = new DefaultFileSystemRule();
    private final PageCacheRule pageCacheRule = new PageCacheRule();
    private final BatchImporterFactory batchImporterFactory = BatchImporterFactory.withHighestPriority();

    @Rule
    public RuleChain ruleChain = RuleChain.outerRule( directory )
                                          .around( fileSystemRule ).around( pageCacheRule );

    @Parameterized.Parameter
    public String version;

    @Parameters( name = "{0}" )
    public static Collection<String> versions()
    {
        return Collections.singletonList( StandardV3_4.STORE_VERSION );
    }

    private final FileSystemAbstraction fs = fileSystemRule.get();
    private JobScheduler jobScheduler;
    private Neo4jLayout neo4jLayout;
    private DatabaseLayout workingDatabaseLayout;
    private Path prepareDirectory;
    private LegacyTransactionLogsLocator legacyTransactionLogsLocator;
    private PageCache pageCache;
    private RecordFormats baselineFormat;
    private RecordFormats successorFormat;

    @Before
    public void setUpLabelScanStore()
    {
        jobScheduler = new ThreadPoolJobScheduler();
        neo4jLayout = Neo4jLayout.of( directory.homePath() );
        workingDatabaseLayout = neo4jLayout.databaseLayout( DEFAULT_DATABASE_NAME );
        prepareDirectory = directory.directoryPath( "prepare" );
        legacyTransactionLogsLocator = new LegacyTransactionLogsLocator( Config.defaults(), workingDatabaseLayout );
        pageCache = pageCacheRule.getPageCache( fs );
        baselineFormat = RecordFormatSelector.selectForVersion( version );
        successorFormat = RecordFormatSelector.findSuccessor( baselineFormat ).orElse( baselineFormat );
    }

    @After
    public void tearDown() throws Exception
    {
        jobScheduler.close();
    }

    @Test
    public void shouldSucceedWithUpgradeAfterPreviousAttemptDiedDuringMigration()
            throws IOException, ConsistencyCheckIncompleteException
    {
        MigrationTestUtils.prepareSampleLegacyDatabase( version, fs, workingDatabaseLayout.databaseDirectory(), prepareDirectory );
        RecordStoreVersionCheck versionCheck = new RecordStoreVersionCheck( fs, pageCache, workingDatabaseLayout, NullLogProvider.getInstance(),
                Config.defaults(), NULL );
        MigrationProgressMonitor progressMonitor = MigrationProgressMonitor.SILENT;
        LogService logService = NullLogService.getInstance();
        RecordStorageMigrator failingStoreMigrator = new RecordStorageMigrator( fs, pageCache, CONFIG, logService, jobScheduler, NULL, batchImporterFactory,
                INSTANCE )
        {
            @Override
            public void migrate( DatabaseLayout directoryLayout, DatabaseLayout migrationLayout,
                    ProgressReporter progressReporter,
                    String versionToMigrateFrom, String versionToMigrateTo ) throws IOException, KernelException
            {
                super.migrate( directoryLayout, migrationLayout, progressReporter, versionToMigrateFrom,
                        versionToMigrateTo );
                throw new RuntimeException( "This upgrade is failing" );
            }
        };

        try
        {
            newUpgrader( versionCheck, progressMonitor, createIndexMigrator(), failingStoreMigrator )
                    .migrateIfNeeded( workingDatabaseLayout, false );
            fail( "Should throw exception" );
        }
        catch ( RuntimeException e )
        {
            assertEquals( "This upgrade is failing", e.getMessage() );
        }

        assertTrue( checkNeoStoreHasFormatVersion( versionCheck, baselineFormat ) );

        RecordStorageMigrator migrator = new RecordStorageMigrator( fs, pageCache, CONFIG, logService, jobScheduler, NULL, batchImporterFactory, INSTANCE );
        IdGeneratorMigrator idMigrator = new IdGeneratorMigrator( fs, pageCache, CONFIG, NULL );
        SchemaIndexMigrator indexMigrator = createIndexMigrator();
        newUpgrader( versionCheck, progressMonitor, indexMigrator, migrator, idMigrator ).migrateIfNeeded( workingDatabaseLayout, false );

        assertTrue( checkNeoStoreHasFormatVersion( versionCheck, successorFormat ) );

        // Since consistency checker is in read only mode we need to start/stop db to generate label scan store.
        startStopDatabase( neo4jLayout.homeDirectory() );
        assertConsistentStore( workingDatabaseLayout );
    }

    private SchemaIndexMigrator createIndexMigrator()
    {
        return new SchemaIndexMigrator( "upgrade test indexes", fs, IndexProvider.EMPTY.directoryStructure(), selectStorageEngine() );
    }

    @Test
    public void tracePageCacheAccessOnIdStoreUpgrade() throws IOException, ConsistencyCheckIncompleteException
    {
        MigrationTestUtils.prepareSampleLegacyDatabase( version, fs, workingDatabaseLayout.databaseDirectory(), prepareDirectory );
        RecordStoreVersionCheck versionCheck = new RecordStoreVersionCheck( fs, pageCache, workingDatabaseLayout, NullLogProvider.getInstance(),
                Config.defaults(), NULL );
        MigrationProgressMonitor progressMonitor = MigrationProgressMonitor.SILENT;
        LogService logService = NullLogService.getInstance();
        var idMigratorTracer = new DefaultPageCacheTracer();
        var recordMigratorTracer = new DefaultPageCacheTracer();
        IdGeneratorMigrator idMigrator = new IdGeneratorMigrator( fs, pageCache, CONFIG, idMigratorTracer );

        assertTrue( checkNeoStoreHasFormatVersion( versionCheck, baselineFormat ) );

        var migrator = new RecordStorageMigrator( fs, pageCache, CONFIG, logService, jobScheduler, recordMigratorTracer, batchImporterFactory, INSTANCE );
        newUpgrader( versionCheck, progressMonitor, createIndexMigrator(), migrator, idMigrator ).migrateIfNeeded( workingDatabaseLayout, false );

        assertTrue( checkNeoStoreHasFormatVersion( versionCheck, successorFormat ) );

        startStopDatabase( neo4jLayout.homeDirectory() );
        assertConsistentStore( workingDatabaseLayout );

        assertEquals( 21, idMigratorTracer.faults() );
        assertEquals( 178, idMigratorTracer.hits() );
        assertEquals( 199, idMigratorTracer.pins() );
        assertEquals( 199, idMigratorTracer.unpins() );

        assertEquals( 52, recordMigratorTracer.faults() );
        assertEquals( 208, recordMigratorTracer.hits() );
        assertEquals( 260, recordMigratorTracer.pins() );
        assertEquals( 260, recordMigratorTracer.unpins() );
    }

    @Test
    public void shouldSucceedWithUpgradeAfterPreviousAttemptDiedDuringMovingFiles()
            throws IOException, ConsistencyCheckIncompleteException
    {
        MigrationTestUtils.prepareSampleLegacyDatabase( version, fs, workingDatabaseLayout.databaseDirectory(), prepareDirectory );
        RecordStoreVersionCheck versionCheck = new RecordStoreVersionCheck( fs, pageCache, workingDatabaseLayout, NullLogProvider.getInstance(),
                Config.defaults(), NULL );
        MigrationProgressMonitor progressMonitor = MigrationProgressMonitor.SILENT;
        LogService logService = NullLogService.getInstance();
        RecordStorageMigrator failingStoreMigrator = new RecordStorageMigrator( fs, pageCache, CONFIG, logService, jobScheduler, NULL, batchImporterFactory,
                INSTANCE )
        {
            @Override
            public void moveMigratedFiles( DatabaseLayout migrationLayout, DatabaseLayout directoryLayout, String versionToUpgradeFrom,
                    String versionToMigrateTo ) throws IOException
            {
                super.moveMigratedFiles( migrationLayout, directoryLayout, versionToUpgradeFrom, versionToMigrateTo );
                throw new RuntimeException( "This upgrade is failing" );
            }
        };
        IdGeneratorMigrator idMigrator = new IdGeneratorMigrator( fs, pageCache, CONFIG, NULL );

        assertTrue( checkNeoStoreHasFormatVersion( versionCheck, baselineFormat ) );

        try
        {
            newUpgrader( versionCheck, progressMonitor, createIndexMigrator(), failingStoreMigrator, idMigrator )
                    .migrateIfNeeded( workingDatabaseLayout, false );
            fail( "Should throw exception" );
        }
        catch ( RuntimeException e )
        {
            assertEquals( "This upgrade is failing", e.getMessage() );
        }

        RecordStorageMigrator migrator = new RecordStorageMigrator( fs, pageCache, CONFIG, logService, jobScheduler, NULL, batchImporterFactory, INSTANCE );
        newUpgrader( versionCheck, progressMonitor, createIndexMigrator(), migrator, idMigrator ).migrateIfNeeded( workingDatabaseLayout, false );

        assertTrue( checkNeoStoreHasFormatVersion( versionCheck, successorFormat ) );

        pageCache.close();

        // Since consistency checker is in read only mode we need to start/stop db to generate label scan store.
        startStopDatabase( neo4jLayout.homeDirectory() );
        assertConsistentStore( workingDatabaseLayout );
    }

    private StoreUpgrader newUpgrader( StoreVersionCheck versionCheck, MigrationProgressMonitor progressMonitor, StoreMigrationParticipant... participants )
    {
        Config config = Config.defaults( allow_upgrade, true );

        Dependencies dependencies = new Dependencies();
        dependencies.satisfyDependencies( new Monitors() );
        RecordStorageEngineFactory storageEngineFactory = new RecordStorageEngineFactory();
        LogsUpgrader logsUpgrader = new LogsUpgrader(
                fs, storageEngineFactory, workingDatabaseLayout, pageCache, legacyTransactionLogsLocator, config, dependencies, NULL, INSTANCE, false );
        StoreUpgrader upgrader = new StoreUpgrader(
                versionCheck, progressMonitor, config, fs, NullLogProvider.getInstance(), logsUpgrader, NULL );
        for ( StoreMigrationParticipant participant : participants )
        {
            upgrader.addParticipant( participant );
        }
        return upgrader;
    }

    private static void startStopDatabase( Path storeDir )
    {
        DatabaseManagementService managementService = new TestDatabaseManagementServiceBuilder( storeDir ).setConfig( allow_upgrade, true ).build();
        managementService.database( DEFAULT_DATABASE_NAME );
        managementService.shutdown();
    }
}
