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
package org.neo4j.kernel.impl.index.schema;

import java.util.Collection;

import org.neo4j.index.internal.gbptree.Seeker;
import org.neo4j.internal.kernel.api.IndexQuery;
import org.neo4j.values.storable.Value;

class FilteringNativeHitIterator<KEY extends NativeIndexKey<KEY>, VALUE extends NativeIndexValue> extends NativeHitIterator<KEY,VALUE>
{
    private final IndexQuery[] filters;

    FilteringNativeHitIterator( Seeker<KEY,VALUE> seeker, Collection<Seeker<KEY,VALUE>> toRemoveFromWhenExhausted, IndexQuery[] filters )
    {
        super( seeker, toRemoveFromWhenExhausted );
        this.filters = filters;
    }

    @Override
    boolean acceptValues( Value[] values )
    {
        for ( int i = 0; i < values.length; i++ )
        {
            if ( !filters[i].acceptsValue( values[i] ) )
            {
                return false;
            }
        }
        return true;
    }
}
