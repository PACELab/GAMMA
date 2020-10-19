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

import org.neo4j.consistency.RecordType;
import org.neo4j.consistency.store.RecordAccess;
import org.neo4j.consistency.store.RecordReference;
import org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer;
import org.neo4j.kernel.impl.store.record.DynamicRecord;

public enum DynamicStore
{
    STRING( RecordType.STRING_PROPERTY )
    {
        @Override
        RecordReference<DynamicRecord> lookup( RecordAccess records, long block, PageCursorTracer cursorTracer )
        {
            return records.string( block, cursorTracer );
        }
    },
    ARRAY( RecordType.ARRAY_PROPERTY )
    {
        @Override
        RecordReference<DynamicRecord> lookup( RecordAccess records, long block, PageCursorTracer cursorTracer )
        {
            return records.array( block, cursorTracer );
        }
    },
    PROPERTY_KEY( RecordType.PROPERTY_KEY_NAME )
    {
        @Override
        RecordReference<DynamicRecord> lookup( RecordAccess records, long block, PageCursorTracer cursorTracer )
        {
            return records.propertyKeyName( (int) block, cursorTracer );
        }
    },
    RELATIONSHIP_TYPE( RecordType.RELATIONSHIP_TYPE_NAME )
    {
        @Override
        RecordReference<DynamicRecord> lookup( RecordAccess records, long block, PageCursorTracer cursorTracer )
        {
            return records.relationshipTypeName( (int) block, cursorTracer );
        }
    },
    LABEL( RecordType.LABEL_NAME )
    {
        @Override
        RecordReference<DynamicRecord> lookup( RecordAccess records, long block, PageCursorTracer cursorTracer )
        {
            return records.labelName( (int) block, cursorTracer );
        }
    },
    NODE_LABEL( RecordType.NODE_DYNAMIC_LABEL )
    {
        @Override
        RecordReference<DynamicRecord> lookup( RecordAccess records, long block, PageCursorTracer cursorTracer )
        {
            return records.nodeLabels( block, cursorTracer );
        }
    };

    public final RecordType type;

    DynamicStore( RecordType type )
    {
        this.type = type;
    }

    abstract RecordReference<DynamicRecord> lookup( RecordAccess records, long block, PageCursorTracer cursorTracer );
}
