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
package org.neo4j.consistency.checking.full;

import org.eclipse.collections.api.iterator.IntIterator;
import org.eclipse.collections.api.map.primitive.MutableIntObjectMap;
import org.eclipse.collections.api.set.primitive.MutableIntSet;
import org.eclipse.collections.impl.map.mutable.primitive.IntObjectHashMap;
import org.eclipse.collections.impl.set.mutable.primitive.IntHashSet;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Function;

import org.neo4j.consistency.RecordType;
import org.neo4j.consistency.report.ConsistencyReport;
import org.neo4j.consistency.report.ConsistencyReporter;
import org.neo4j.internal.recordstorage.SchemaRuleAccess;
import org.neo4j.internal.recordstorage.StoreTokens;
import org.neo4j.internal.schema.ConstraintDescriptor;
import org.neo4j.internal.schema.LabelSchemaDescriptor;
import org.neo4j.internal.schema.RelationTypeSchemaDescriptor;
import org.neo4j.internal.schema.SchemaDescriptor;
import org.neo4j.internal.schema.SchemaProcessor;
import org.neo4j.io.pagecache.tracing.PageCacheTracer;
import org.neo4j.io.pagecache.tracing.cursor.PageCursorTracer;
import org.neo4j.kernel.impl.store.StoreAccess;
import org.neo4j.kernel.impl.store.record.NodeRecord;
import org.neo4j.kernel.impl.store.record.PrimitiveRecord;
import org.neo4j.kernel.impl.store.record.RelationshipRecord;
import org.neo4j.token.TokenHolders;

import static org.neo4j.internal.helpers.Numbers.safeCastLongToInt;

public class MandatoryProperties
{
    private static final String MANDATORY_PROPERTIES_CHECKER_TAG = "mandatoryPropertiesChecker";
    private final MutableIntObjectMap<int[]> nodes = new IntObjectHashMap<>();
    private final MutableIntObjectMap<int[]> relationships = new IntObjectHashMap<>();
    private final StoreAccess storeAccess;

    public MandatoryProperties( StoreAccess storeAccess, PageCacheTracer pageCacheTracer )
    {
        this.storeAccess = storeAccess;
        try ( var cursorTracer = pageCacheTracer.createPageCursorTracer( MANDATORY_PROPERTIES_CHECKER_TAG ) )
        {
            TokenHolders tokenHolders = StoreTokens.readOnlyTokenHolders( storeAccess.getRawNeoStores(), cursorTracer );
            SchemaRuleAccess schemaRuleAccess = SchemaRuleAccess.getSchemaRuleAccess( storeAccess.getSchemaStore(), tokenHolders );
            for ( ConstraintDescriptor constraint : constraintsIgnoringMalformed( schemaRuleAccess, cursorTracer ) )
            {
                if ( constraint.enforcesPropertyExistence() )
                {
                    SchemaProcessor constraintRecorder = new MandatoryPropertiesSchemaProcessor();
                    constraint.schema().processWith( constraintRecorder );
                }
            }
        }
    }

    public BiFunction<NodeRecord,PageCursorTracer,Check<NodeRecord,ConsistencyReport.NodeConsistencyReport>> forNodes(
            final ConsistencyReporter reporter )
    {
        return ( node, cursorTracer ) ->
        {
            MutableIntSet keys = null;
            for ( long labelId : NodeLabelReader.getListOfLabels( node, storeAccess.getNodeDynamicLabelStore(), cursorTracer ) )
            {
                // labelId _is_ actually an int. A technical detail in the store format has these come in a long[]
                int[] propertyKeys = nodes.get( safeCastLongToInt( labelId ) );
                if ( propertyKeys != null )
                {
                    if ( keys == null )
                    {
                        keys = new IntHashSet( 16 );
                    }
                    for ( int key : propertyKeys )
                    {
                        keys.add( key );
                    }
                }
            }
            return keys != null
                    ? new RealCheck<>( node, ConsistencyReport.NodeConsistencyReport.class, reporter, RecordType.NODE, keys )
                    : MandatoryProperties.noCheck();
        };
    }

    public Function<RelationshipRecord,Check<RelationshipRecord,ConsistencyReport.RelationshipConsistencyReport>>
            forRelationships( final ConsistencyReporter reporter )
    {
        return relationship ->
        {
            int[] propertyKeys = relationships.get( relationship.getType() );
            if ( propertyKeys != null )
            {
                final MutableIntSet keys = new IntHashSet( propertyKeys.length );
                for ( int key : propertyKeys )
                {
                    keys.add( key );
                }
                return new RealCheck<>( relationship, ConsistencyReport.RelationshipConsistencyReport.class,
                        reporter, RecordType.RELATIONSHIP, keys );
            }
            return noCheck();
        };
    }

    private Iterable<ConstraintDescriptor> constraintsIgnoringMalformed( SchemaRuleAccess schemaStorage, PageCursorTracer cursorTracer )
    {
        return () -> schemaStorage.constraintsGetAllIgnoreMalformed( cursorTracer );
    }

    private static void recordConstraint( int labelOrRelType, int propertyKey, MutableIntObjectMap<int[]> storage )
    {
        int[] propertyKeys = storage.get( labelOrRelType );
        if ( propertyKeys == null )
        {
            propertyKeys = new int[]{propertyKey};
        }
        else
        {
            propertyKeys = Arrays.copyOf( propertyKeys, propertyKeys.length + 1 );
            propertyKeys[propertyKeys.length - 1] = propertyKey;
        }
        storage.put( labelOrRelType, propertyKeys );
    }

    public interface Check<RECORD extends PrimitiveRecord,REPORT extends ConsistencyReport.PrimitiveConsistencyReport>
            extends AutoCloseable
    {
        void receive( int[] keys );

        @Override
        void close();
    }

    @SuppressWarnings( "unchecked" )
    private static <RECORD extends PrimitiveRecord,
            REPORT extends ConsistencyReport.PrimitiveConsistencyReport> Check<RECORD,REPORT> noCheck()
    {
        return NONE;
    }

    @SuppressWarnings( "rawtypes" )
    private static final Check NONE = new Check()
    {
        @Override
        public void receive( int[] keys )
        {
        }

        @Override
        public void close()
        {
        }

        @Override
        public String toString()
        {
            return "NONE";
        }
    };

    private static class RealCheck<RECORD extends PrimitiveRecord,REPORT extends ConsistencyReport.PrimitiveConsistencyReport>
            implements Check<RECORD,REPORT>
    {
        private final RECORD record;
        private final MutableIntSet mandatoryKeys;
        private final Class<REPORT> reportClass;
        private final ConsistencyReporter reporter;
        private final RecordType recordType;

        RealCheck( RECORD record, Class<REPORT> reportClass, ConsistencyReporter reporter, RecordType recordType, MutableIntSet mandatoryKeys )
        {
            this.record = record;
            this.reportClass = reportClass;
            this.reporter = reporter;
            this.recordType = recordType;
            this.mandatoryKeys = mandatoryKeys;
        }

        @Override
        public void receive( int[] keys )
        {
            for ( int key : keys )
            {
                mandatoryKeys.remove( key );
            }
        }

        @Override
        public void close()
        {
            if ( !mandatoryKeys.isEmpty() )
            {
                for ( IntIterator key = mandatoryKeys.intIterator(); key.hasNext(); )
                {
                    reporter.report( record, reportClass, recordType ).missingMandatoryProperty( key.next() );
                }
            }
        }

        @Override
        public String toString()
        {
            return "Mandatory properties: " + mandatoryKeys;
        }
    }

    private class MandatoryPropertiesSchemaProcessor implements SchemaProcessor
    {
        @Override
        public void processSpecific( LabelSchemaDescriptor schema )
        {
            for ( int propertyId : schema.getPropertyIds() )
            {
                recordConstraint( schema.getLabelId(), propertyId, nodes );
            }
        }

        @Override
        public void processSpecific( RelationTypeSchemaDescriptor schema )
        {
            for ( int propertyId : schema.getPropertyIds() )
            {
                recordConstraint( schema.getRelTypeId(), propertyId, relationships );
            }
        }

        @Override
        public void processSpecific( SchemaDescriptor schema )
        {
            throw new IllegalStateException( "General SchemaDescriptors cannot support constraints" );
        }
    }
}
