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
package org.neo4j.internal.batchimport.cache.idmapping.string;

import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameter;
import org.junit.runners.Parameterized.Parameters;

import java.util.Arrays;
import java.util.Collection;
import java.util.function.Function;

import org.neo4j.internal.batchimport.cache.NumberArrayFactory;
import org.neo4j.internal.batchimport.cache.PageCachedNumberArrayFactory;
import org.neo4j.io.pagecache.tracing.PageCacheTracer;
import org.neo4j.test.rule.PageCacheAndDependenciesRule;
import org.neo4j.test.rule.RandomRule;
import org.neo4j.test.rule.fs.DefaultFileSystemRule;
import org.neo4j.values.storable.RandomValues;

import static org.junit.Assert.assertEquals;
import static org.neo4j.io.pagecache.PageCache.PAGE_SIZE;
import static org.neo4j.memory.EmptyMemoryTracker.INSTANCE;

@RunWith( Parameterized.class )
public class StringCollisionValuesTest
{
    @Rule
    public final PageCacheAndDependenciesRule storage = new PageCacheAndDependenciesRule().with( new DefaultFileSystemRule() );
    @Rule
    public final RandomRule random = new RandomRule().withConfiguration( new RandomValues.Default()
    {
        @Override
        public int stringMaxLength()
        {
            return (1 << Short.SIZE) - 1;
        }
    } );

    @Parameters
    public static Collection<Function<PageCacheAndDependenciesRule,NumberArrayFactory>> data()
    {
        return Arrays.asList(
                storage -> NumberArrayFactory.HEAP,
                storage -> NumberArrayFactory.OFF_HEAP,
                storage -> NumberArrayFactory.AUTO_WITHOUT_PAGECACHE,
                storage -> NumberArrayFactory.CHUNKED_FIXED_SIZE,
                storage -> new PageCachedNumberArrayFactory( storage.pageCache(), PageCacheTracer.NULL, storage.directory().homePath() ) );
    }

    @Parameter( 0 )
    public Function<PageCacheAndDependenciesRule,NumberArrayFactory> factory;

    @Test
    public void shouldStoreAndLoadStrings()
    {
        // given
        try ( StringCollisionValues values = new StringCollisionValues( factory.apply( storage ), 10_000, INSTANCE ) )
        {
            // when
            long[] offsets = new long[100];
            String[] strings = new String[offsets.length];
            for ( int i = 0; i < offsets.length; i++ )
            {
                String string = random.nextAlphaNumericString();
                offsets[i] = values.add( string );
                strings[i] = string;
            }

            // then
            for ( int i = 0; i < offsets.length; i++ )
            {
                assertEquals( strings[i], values.get( offsets[i] ) );
            }
        }
    }

    @Test
    public void shouldMoveOverToNextChunkOnNearEnd()
    {
        // given
        try ( StringCollisionValues values = new StringCollisionValues( factory.apply( storage ), 10_000, INSTANCE ) )
        {
            char[] chars = new char[PAGE_SIZE - 3];
            Arrays.fill( chars, 'a' );

            // when
            String string = String.valueOf( chars );
            long offset = values.add( string );
            String secondString = "abcdef";
            long secondOffset = values.add( secondString );

            // then
            String readString = (String) values.get( offset );
            assertEquals( string, readString );
            String readSecondString = (String) values.get( secondOffset );
            assertEquals( secondString, readSecondString );
        }
    }
}
