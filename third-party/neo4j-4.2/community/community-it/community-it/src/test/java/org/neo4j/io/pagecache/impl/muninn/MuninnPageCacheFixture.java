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
package org.neo4j.io.pagecache.impl.muninn;

import java.util.concurrent.CountDownLatch;

import org.neo4j.configuration.Config;
import org.neo4j.configuration.pagecache.ConfigurableIOBufferFactory;
import org.neo4j.io.mem.MemoryAllocator;
import org.neo4j.io.pagecache.PageCacheTestSupport;
import org.neo4j.io.pagecache.PageSwapperFactory;
import org.neo4j.io.pagecache.buffer.IOBufferFactory;
import org.neo4j.io.pagecache.tracing.PageCacheTracer;
import org.neo4j.io.pagecache.tracing.cursor.context.VersionContextSupplier;
import org.neo4j.memory.LocalMemoryTracker;
import org.neo4j.scheduler.JobScheduler;
import org.neo4j.time.Clocks;

public class MuninnPageCacheFixture extends PageCacheTestSupport.Fixture<MuninnPageCache>
{
    CountDownLatch backgroundFlushLatch;
    private MemoryAllocator allocator;

    @Override
    public MuninnPageCache createPageCache( PageSwapperFactory swapperFactory, int maxPages, PageCacheTracer tracer, VersionContextSupplier contextSupplier,
            JobScheduler jobScheduler, IOBufferFactory bufferFactory )
    {
        long memory = MuninnPageCache.memoryRequiredForPages( maxPages );
        var memoryTracker = new LocalMemoryTracker();
        allocator = MemoryAllocator.createAllocator( memory, memoryTracker );
        var usedBufferFactory = selectBufferFactory( bufferFactory, memoryTracker );
        return new MuninnPageCache( swapperFactory, allocator, tracer, contextSupplier, jobScheduler, Clocks.nanoClock(), memoryTracker, usedBufferFactory );
    }

    private static IOBufferFactory selectBufferFactory( IOBufferFactory bufferFactory, LocalMemoryTracker memoryTracker )
    {
        return bufferFactory != null ? bufferFactory : new ConfigurableIOBufferFactory( Config.defaults(), memoryTracker );
    }

    @Override
    public void tearDownPageCache( MuninnPageCache pageCache )
    {
        if ( backgroundFlushLatch != null )
        {
            backgroundFlushLatch.countDown();
            backgroundFlushLatch = null;
        }
        pageCache.close();
        allocator.close();
    }
}
