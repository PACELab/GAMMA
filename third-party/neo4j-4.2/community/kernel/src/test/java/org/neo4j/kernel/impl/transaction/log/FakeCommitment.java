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
package org.neo4j.kernel.impl.transaction.log;

import org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer;
import org.neo4j.storageengine.api.TransactionIdStore;

import static org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer.NULL;

public class FakeCommitment implements Commitment
{
    public static final int CHECKSUM = 3;
    public static final long TIMESTAMP = 8194639457389L;
    private final long id;
    private final TransactionIdStore transactionIdStore;
    private boolean committed;

    public FakeCommitment( long id, TransactionIdStore transactionIdStore )
    {
        this( id, transactionIdStore, false );
    }

    public FakeCommitment( long id, TransactionIdStore transactionIdStore, boolean markedAsCommitted )
    {
        this.id = id;
        this.transactionIdStore = transactionIdStore;
        this.committed = markedAsCommitted;
    }

    @Override
    public void publishAsCommitted( PageCursorTracer cursorTracer )
    {
        committed = true;
        transactionIdStore.transactionCommitted( id, CHECKSUM, TIMESTAMP, NULL );
    }

    @Override
    public void publishAsClosed( PageCursorTracer cursorTracer )
    {
        transactionIdStore.transactionClosed( id, 1, 2, cursorTracer );
    }

    @Override
    public boolean markedAsCommitted()
    {
        return committed;
    }

}
