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
package org.neo4j.kernel.impl.scheduler;

import java.util.LinkedList;
import java.util.Queue;
import javax.annotation.Nonnull;

import org.neo4j.scheduler.DeferredExecutor;
import org.neo4j.scheduler.JobMonitoringParams;
import org.neo4j.scheduler.MonitoredJobExecutor;

/**
 * Buffers all tasks sent to it, and is able to replay those messages into
 * another Executor.
 * <p>
 * This will replay tasks in the order they are received.
 * <p>
 * You should also not use this executor, when there is a risk that it can be
 * subjected to an unbounded quantity of tasks, since the buffer keeps
 * all messages until it gets a chance to replay them.
 */
public class BufferingExecutor implements DeferredExecutor
{
    private final Queue<Command> buffer = new LinkedList<>();

    private volatile MonitoredJobExecutor realExecutor;

    @Override
    public void satisfyWith( MonitoredJobExecutor executor )
    {
        synchronized ( this )
        {
            if ( realExecutor != null )
            {
                throw new RuntimeException( "real executor is already set. Cannot override" );
            }
            realExecutor = executor;
            replayBuffer();
        }
    }

    private void replayBuffer()
    {
        Command command = pollCommand();
        while ( command != null )
        {
            realExecutor.execute( command.monitoringParams, command.runnable );
            command = pollCommand();
        }
    }

    private Command pollCommand()
    {
        synchronized ( buffer )
        {
            return buffer.poll();
        }
    }

    private void queueCommand( Command command )
    {
        synchronized ( buffer )
        {
            buffer.add( command );
        }
    }

    @Override
    public void execute( @Nonnull JobMonitoringParams monitoringParams, @Nonnull Runnable command )
    {
        // First do an unsynchronized check to see if a realExecutor is present
        if ( realExecutor != null )
        {
            realExecutor.execute( monitoringParams, command );
            return;
        }

        // Now do a synchronized check to avoid race conditions
        synchronized ( this )
        {
            if ( realExecutor != null )
            {
                realExecutor.execute( monitoringParams, command );
            }
            else
            {
                queueCommand( new Command( monitoringParams, command ) );
            }
        }
    }

    private static class Command
    {
        private final JobMonitoringParams monitoringParams;
        private final Runnable runnable;

        Command( JobMonitoringParams monitoringParams, Runnable runnable )
        {
            this.monitoringParams = monitoringParams;
            this.runnable = runnable;
        }
    }
}
