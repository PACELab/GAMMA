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
package org.neo4j.internal.recordstorage;

import java.io.IOException;
import java.util.Iterator;

import org.neo4j.internal.helpers.collection.Iterators;
import org.neo4j.internal.helpers.collection.Visitor;
import org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer;
import org.neo4j.storageengine.api.CommandsToApply;
import org.neo4j.storageengine.api.StorageCommand;
import org.neo4j.common.Subject;
import org.neo4j.storageengine.api.TransactionIdStore;

import static org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer.NULL;

public class GroupOfCommands implements CommandsToApply
{
    private final long transactionId;
    private final StorageCommand[] commands;
    GroupOfCommands next;

    public GroupOfCommands( StorageCommand... commands )
    {
        this( TransactionIdStore.BASE_TX_ID, commands );
    }

    public GroupOfCommands( long transactionId, StorageCommand... commands )
    {
        this.transactionId = transactionId;
        this.commands = commands;
    }

    @Override
    public long transactionId()
    {
        return transactionId;
    }

    @Override
    public Subject subject()
    {
        return Subject.SYSTEM;
    }

    @Override
    public PageCursorTracer cursorTracer()
    {
        return NULL;
    }

    @Override
    public CommandsToApply next()
    {
        return next;
    }

    @Override
    public boolean accept( Visitor<StorageCommand,IOException> visitor ) throws IOException
    {
        for ( StorageCommand command : commands )
        {
            if ( visitor.visit( command ) )
            {
                return true;
            }
        }
        return false;
    }

    @Override
    public Iterator<StorageCommand> iterator()
    {
        return Iterators.iterator( commands );
    }
}
