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
package org.neo4j.kernel.builtinprocs;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.stream.Collectors;

import org.neo4j.collection.RawIterator;
import org.neo4j.configuration.GraphDatabaseSettings;
import org.neo4j.exceptions.KernelException;
import org.neo4j.internal.kernel.api.exceptions.ProcedureException;
import org.neo4j.internal.kernel.api.procs.QualifiedName;
import org.neo4j.kernel.impl.api.integrationtest.KernelIntegrationTest;
import org.neo4j.kernel.impl.util.DefaultValueMapper;
import org.neo4j.values.AnyValue;
import org.neo4j.values.storable.BooleanValue;
import org.neo4j.values.storable.TextValue;
import org.neo4j.values.storable.Values;

import static org.apache.commons.lang3.ArrayUtils.toArray;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.neo4j.configuration.SettingValueParsers.FALSE;
import static org.neo4j.internal.helpers.collection.Iterators.asList;
import static org.neo4j.internal.kernel.api.procs.ProcedureSignature.procedureName;
import static org.neo4j.internal.kernel.api.security.SecurityContext.AUTH_DISABLED;
import static org.neo4j.kernel.api.ResourceTracker.EMPTY_RESOURCE_TRACKER;
import static org.neo4j.values.storable.Values.stringValue;

class BuiltInDbmsProceduresIT extends KernelIntegrationTest
{
    @Test
    void listConfig() throws Exception
    {
        // When
        List<AnyValue[]> config = callListConfig( "" );
        List<String> names = config.stream()
                .map( o -> ((TextValue) o[0]).stringValue() )
                .collect( Collectors.toList() );

        // The size of the config is not fixed so just make sure it's the right magnitude
        assertTrue( names.size() > 10 );

        assertThat( names ).contains( GraphDatabaseSettings.record_format.name() );

        // Should not contain "unsupported.*" configs
        assertEquals( 0, names.stream()
                .filter( n -> n.startsWith( "unsupported" ) )
                .count() );
    }

    @Test
    void listConfigWithASpecificConfigName() throws Exception
    {
        // When
        List<AnyValue[]> config = callListConfig( GraphDatabaseSettings.strict_config_validation.name() );

        assertEquals( 1, config.size() );
        assertArrayEquals( new AnyValue[]{stringValue( "dbms.config.strict_validation" ),
                stringValue(
                        "A strict configuration validation will prevent the database from starting up if unknown " +
                        "configuration options are specified in the neo4j settings namespace (such as dbms., " +
                        "cypher., etc)." ),
                stringValue( FALSE ), Values.FALSE}, config.get( 0 ) );
    }

    @Test
    void listClientConfig() throws Exception
    {
        QualifiedName procedureName = procedureName( "dbms", "clientConfig" );
        int procedureId = procs().procedureGet( procedureName ).id();
        RawIterator<AnyValue[],ProcedureException> callResult =
                dbmsOperations()
                        .procedureCallDbms( procedureId, new AnyValue[]{}, transaction, dependencyResolver,
                                            AUTH_DISABLED, EMPTY_RESOURCE_TRACKER, new DefaultValueMapper( transaction ) );
        List<AnyValue[]> config = asList( callResult );
        assertEquals( 4, config.size());

        assertEquals( config.get( 0 )[0], stringValue( "browser.post_connect_cmd" ));
        assertEquals( config.get( 1 )[0], stringValue( "browser.remote_content_hostname_whitelist" ));
        assertEquals( config.get( 2 )[0], stringValue( "dbms.default_database" ));
        assertEquals( config.get( 3 )[0], stringValue( "dbms.security.auth_enabled" ));

    }

    @Test
    void durationAlwaysListedWithUnit() throws Exception
    {
        // When
        List<AnyValue[]> config = callListConfig( GraphDatabaseSettings.transaction_timeout.name() );

        assertEquals( 1, config.size() );
        assertArrayEquals( new AnyValue[]{ stringValue( "dbms.transaction.timeout" ),
                stringValue( "The maximum time interval of a transaction within which it should be completed." ),
                stringValue( "0s" ), Values.TRUE }, config.get( 0 ) );
    }

    @Test
    void listDynamicSetting() throws KernelException
    {
        List<AnyValue[]> config = callListConfig( GraphDatabaseSettings.check_point_iops_limit.name() );

        assertEquals( 1, config.size() );
        assertTrue( ((BooleanValue) config.get(0)[3]).booleanValue() );
    }

    @Test
    void listNotDynamicSetting() throws KernelException
    {
        List<AnyValue[]> config = callListConfig( GraphDatabaseSettings.data_directory.name() );

        assertEquals( 1, config.size() );
        assertFalse(((BooleanValue) config.get(0)[3]).booleanValue() );
    }

    private List<AnyValue[]> callListConfig( String searchString ) throws KernelException
    {
        QualifiedName procedureName = procedureName( "dbms", "listConfig" );
        int procedureId = procs().procedureGet( procedureName ).id();
        RawIterator<AnyValue[],ProcedureException> callResult =
                dbmsOperations()
                        .procedureCallDbms( procedureId, toArray( stringValue( searchString ) ), transaction, dependencyResolver,
                                AUTH_DISABLED, EMPTY_RESOURCE_TRACKER, new DefaultValueMapper( transaction ) );
        return asList( callResult );
    }
}
