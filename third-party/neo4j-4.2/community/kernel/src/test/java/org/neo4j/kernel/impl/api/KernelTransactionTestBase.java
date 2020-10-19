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
package org.neo4j.kernel.impl.api;

import org.eclipse.collections.api.map.primitive.MutableLongObjectMap;
import org.eclipse.collections.api.set.primitive.MutableLongSet;
import org.eclipse.collections.impl.map.mutable.primitive.LongObjectHashMap;
import org.junit.jupiter.api.BeforeEach;
import org.mockito.Mockito;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import org.neo4j.collection.Dependencies;
import org.neo4j.collection.pool.Pool;
import org.neo4j.configuration.Config;
import org.neo4j.configuration.GraphDatabaseSettings;
import org.neo4j.internal.index.label.LabelScanStore;
import org.neo4j.internal.index.label.RelationshipTypeScanStore;
import org.neo4j.internal.kernel.api.security.LoginContext;
import org.neo4j.internal.kernel.api.security.SecurityContext;
import org.neo4j.internal.schema.SchemaState;
import org.neo4j.io.ByteUnit;
import org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer;
import org.neo4j.io.pagecache.tracing.cursor.context.EmptyVersionContextSupplier;
import org.neo4j.kernel.api.KernelTransaction;
import org.neo4j.kernel.availability.AvailabilityGuard;
import org.neo4j.kernel.database.DatabaseTracers;
import org.neo4j.kernel.database.TestDatabaseIdRepository;
import org.neo4j.kernel.impl.api.index.IndexingService;
import org.neo4j.kernel.impl.api.index.stats.IndexStatisticsStore;
import org.neo4j.kernel.impl.constraints.StandardConstraintSemantics;
import org.neo4j.kernel.impl.factory.CanWrite;
import org.neo4j.kernel.impl.factory.GraphDatabaseFacade;
import org.neo4j.kernel.impl.locking.Locks;
import org.neo4j.kernel.impl.locking.NoOpClient;
import org.neo4j.kernel.impl.locking.SimpleStatementLocks;
import org.neo4j.kernel.impl.locking.StatementLocks;
import org.neo4j.kernel.impl.transaction.TransactionMonitor;
import org.neo4j.kernel.impl.transaction.TransactionRepresentation;
import org.neo4j.kernel.impl.transaction.tracing.CommitEvent;
import org.neo4j.kernel.impl.util.collection.CollectionsFactory;
import org.neo4j.kernel.impl.util.collection.OnHeapCollectionsFactory;
import org.neo4j.kernel.impl.util.diffsets.MutableLongDiffSets;
import org.neo4j.kernel.internal.event.DatabaseTransactionEventListeners;
import org.neo4j.lock.ResourceLocker;
import org.neo4j.memory.MemoryGroup;
import org.neo4j.memory.MemoryPools;
import org.neo4j.memory.MemoryTracker;
import org.neo4j.resources.CpuClock;
import org.neo4j.storageengine.api.CommandCreationContext;
import org.neo4j.storageengine.api.MetadataProvider;
import org.neo4j.storageengine.api.StorageCommand;
import org.neo4j.storageengine.api.StorageEngine;
import org.neo4j.storageengine.api.StorageReader;
import org.neo4j.storageengine.api.TransactionApplicationMode;
import org.neo4j.storageengine.api.TransactionIdStore;
import org.neo4j.storageengine.api.txstate.ReadableTransactionState;
import org.neo4j.storageengine.api.txstate.TxStateVisitor;
import org.neo4j.time.Clocks;
import org.neo4j.time.FakeClock;
import org.neo4j.token.TokenHolders;
import org.neo4j.token.api.TokenHolder;
import org.neo4j.values.storable.Value;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyCollection;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.RETURNS_MOCKS;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.neo4j.configuration.GraphDatabaseSettings.DEFAULT_DATABASE_NAME;
import static org.neo4j.internal.kernel.api.connectioninfo.ClientConnectionInfo.EMBEDDED_CONNECTION;
import static org.neo4j.internal.kernel.api.security.LoginContext.AUTH_DISABLED;
import static org.neo4j.storageengine.api.TransactionIdStore.BASE_TX_COMMIT_TIMESTAMP;

class KernelTransactionTestBase
{
    protected final StorageEngine storageEngine = mock( StorageEngine.class );
    protected final StorageReader storageReader = mock( StorageReader.class );
    protected final MetadataProvider metadataProvider = mock( MetadataProvider.class );
    protected final CommandCreationContext commandCreationContext = mock( CommandCreationContext.class );
    protected final TransactionMonitor transactionMonitor = mock( TransactionMonitor.class );
    protected final CapturingCommitProcess commitProcess = new CapturingCommitProcess();
    protected final AvailabilityGuard availabilityGuard = mock( AvailabilityGuard.class );
    protected final FakeClock clock = Clocks.fakeClock();
    protected final Pool<KernelTransactionImplementation> txPool = mock( Pool.class );
    protected CollectionsFactory collectionsFactory;

    protected final Config config = Config.defaults();
    private final long defaultTransactionTimeoutMillis = config.get( GraphDatabaseSettings.transaction_timeout ).toMillis();

    @BeforeEach
    public void before() throws Exception
    {
        collectionsFactory = Mockito.spy( new TestCollectionsFactory() );
        when( storageEngine.newReader() ).thenReturn( storageReader );
        when( storageEngine.newCommandCreationContext( any( PageCursorTracer.class ), any() ) ).thenReturn( commandCreationContext );
        when( storageEngine.metadataProvider() ).thenReturn( metadataProvider );
        doAnswer( invocation -> ((Collection<StorageCommand>) invocation.getArgument(0) ).add( new TestCommand() ) )
            .when( storageEngine ).createCommands(
                    anyCollection(),
                    any( ReadableTransactionState.class ),
                    any( StorageReader.class ),
                    any( CommandCreationContext.class ),
                    any( ResourceLocker.class ),
                    anyLong(),
                    any( TxStateVisitor.Decorator.class ), any( PageCursorTracer.class ), any( MemoryTracker.class ) );
    }

    public KernelTransactionImplementation newTransaction( long transactionTimeoutMillis )
    {
        return newTransaction( 0, AUTH_DISABLED, transactionTimeoutMillis );
    }

    public KernelTransactionImplementation newTransaction( LoginContext loginContext )
    {
        return newTransaction( 0, loginContext );
    }

    public KernelTransactionImplementation newTransaction( LoginContext loginContext, Locks.Client locks )
    {
        return newTransaction( 0, loginContext, locks, defaultTransactionTimeoutMillis );
    }

    public KernelTransactionImplementation newTransaction( long lastTransactionIdWhenStarted, LoginContext loginContext )
    {
        return newTransaction( lastTransactionIdWhenStarted, loginContext, defaultTransactionTimeoutMillis );
    }

    public KernelTransactionImplementation newTransaction( long lastTransactionIdWhenStarted, LoginContext loginContext,
            long transactionTimeoutMillis )
    {
        return newTransaction( lastTransactionIdWhenStarted, loginContext, new NoOpClient(), transactionTimeoutMillis );
    }

    public KernelTransactionImplementation newTransaction( long lastTransactionIdWhenStarted, LoginContext loginContext,
            Locks.Client locks, long transactionTimeout )
    {
        KernelTransactionImplementation tx = newNotInitializedTransaction();
        StatementLocks statementLocks = new SimpleStatementLocks( locks );
        SecurityContext securityContext = loginContext.authorize( LoginContext.IdLookup.EMPTY, DEFAULT_DATABASE_NAME );
        tx.initialize( lastTransactionIdWhenStarted, BASE_TX_COMMIT_TIMESTAMP, statementLocks, KernelTransaction.Type.EXPLICIT,
                securityContext, transactionTimeout, 1L, EMBEDDED_CONNECTION );
        return tx;
    }

    KernelTransactionImplementation newNotInitializedTransaction()
    {
        return newNotInitializedTransaction( LeaseService.NO_LEASES );
    }

    KernelTransactionImplementation newNotInitializedTransaction( LeaseService leaseService )
    {
        Dependencies dependencies = new Dependencies();
        dependencies.satisfyDependency( mock( GraphDatabaseFacade.class ) );
        var memoryPool = new MemoryPools().pool( MemoryGroup.TRANSACTION, ByteUnit.mebiBytes( 4 ), null );
        return new KernelTransactionImplementation( config, mock( DatabaseTransactionEventListeners.class ),
                null, null,
                commitProcess, transactionMonitor, txPool, clock, new AtomicReference<>( CpuClock.NOT_AVAILABLE ),
                mock( DatabaseTracers.class, RETURNS_MOCKS ), storageEngine,
                new CanWrite(), EmptyVersionContextSupplier.EMPTY, () -> collectionsFactory,
                new StandardConstraintSemantics(), mock( SchemaState.class ), mockedTokenHolders(),
                mock( IndexingService.class ), mock( LabelScanStore.class ), mock( RelationshipTypeScanStore.class ), mock( IndexStatisticsStore.class ),
                dependencies, new TestDatabaseIdRepository().defaultDatabase(), leaseService, memoryPool );
    }

    public static class CapturingCommitProcess implements TransactionCommitProcess
    {
        private long txId = TransactionIdStore.BASE_TX_ID;
        public List<TransactionRepresentation> transactions = new ArrayList<>();

        @Override
        public long commit( TransactionToApply batch, CommitEvent commitEvent,
                            TransactionApplicationMode mode )
        {
            transactions.add( batch.transactionRepresentation() );
            return ++txId;
        }
    }

    private static TokenHolders mockedTokenHolders()
    {
        return new TokenHolders(
                mock( TokenHolder.class ),
                mock( TokenHolder.class ),
                mock( TokenHolder.class ) );
    }

    private static class TestCollectionsFactory implements CollectionsFactory
    {

        @Override
        public MutableLongSet newLongSet( MemoryTracker memoryTracker )
        {
            return OnHeapCollectionsFactory.INSTANCE.newLongSet( memoryTracker );
        }

        @Override
        public MutableLongDiffSets newLongDiffSets( MemoryTracker memoryTracker )
        {
            return OnHeapCollectionsFactory.INSTANCE.newLongDiffSets( memoryTracker );
        }

        @Override
        public MutableLongObjectMap<Value> newValuesMap( MemoryTracker memoryTracker )
        {
            return new LongObjectHashMap<>();
        }

        @Override
        public void release()
        {
            // nop
        }
    }
}
