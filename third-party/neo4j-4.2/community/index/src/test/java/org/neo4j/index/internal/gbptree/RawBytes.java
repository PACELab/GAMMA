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
package org.neo4j.index.internal.gbptree;

import java.util.Arrays;
import java.util.StringJoiner;

import static org.apache.commons.lang3.ArrayUtils.EMPTY_BYTE_ARRAY;

class RawBytes
{
    static RawBytes EMPTY_BYTES = new RawBytes( EMPTY_BYTE_ARRAY );

    byte[] bytes;

    RawBytes()
    {
        this( EMPTY_BYTE_ARRAY );
    }

    RawBytes( byte[] byteArray )
    {
        bytes = byteArray;
    }

    @Override
    public String toString()
    {
        StringJoiner joiner = new StringJoiner( ", ", "[", "]" );
        int index = 0;
        int nbrOfAccumulatedZeroes = 0;
        while ( index < bytes.length )
        {
            if ( bytes[index] != (byte) 0 )
            {
                if ( nbrOfAccumulatedZeroes > 0 )
                {
                    joiner.add( replaceZeroes( nbrOfAccumulatedZeroes ) );
                    nbrOfAccumulatedZeroes = 0;
                }
                joiner.add( Byte.toString( bytes[index] ) );
            }
            else
            {
                nbrOfAccumulatedZeroes++;
            }
            index++;
        }
        if ( nbrOfAccumulatedZeroes > 0 )
        {
            joiner.add( replaceZeroes( nbrOfAccumulatedZeroes ) );
        }
        return joiner.toString();
    }

    private String replaceZeroes( int nbrOfZeroes )
    {
        if ( nbrOfZeroes > 3 )
        {
            return "0...>" + nbrOfZeroes;
        }
        else
        {
            StringJoiner joiner = new StringJoiner( ", " );
            for ( int i = 0; i < nbrOfZeroes; i++ )
            {
                joiner.add( "0" );
            }
            return joiner.toString();
        }
    }

    void copyFrom( RawBytes source )
    {
        bytes = Arrays.copyOf( source.bytes, source.bytes.length );
    }
}
