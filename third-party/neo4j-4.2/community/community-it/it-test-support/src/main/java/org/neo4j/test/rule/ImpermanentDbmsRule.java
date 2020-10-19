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
package org.neo4j.test.rule;

import org.neo4j.dbms.api.DatabaseManagementServiceBuilder;
import org.neo4j.logging.LogProvider;
import org.neo4j.test.TestDatabaseManagementServiceBuilder;

/**
 * JUnit @Rule for configuring, creating and managing an ImpermanentGraphDatabase instance.
 */
public class ImpermanentDbmsRule extends DbmsRule
{
    private final LogProvider userLogProvider;
    private final LogProvider internalLogProvider;

    public ImpermanentDbmsRule()
    {
        this( null );
    }

    public ImpermanentDbmsRule( LogProvider logProvider )
    {
        this.userLogProvider = logProvider;
        this.internalLogProvider = logProvider;
    }

    @Override
    public ImpermanentDbmsRule startLazily()
    {
        return (ImpermanentDbmsRule) super.startLazily();
    }

    @Override
    protected DatabaseManagementServiceBuilder newFactory()
    {
        return maybeSetInternalLogProvider( maybeSetUserLogProvider( new TestDatabaseManagementServiceBuilder().impermanent() ) );
    }

    protected final TestDatabaseManagementServiceBuilder maybeSetUserLogProvider( TestDatabaseManagementServiceBuilder factory )
    {
        return ( userLogProvider == null ) ? factory : factory.setUserLogProvider( userLogProvider );
    }

    protected final TestDatabaseManagementServiceBuilder maybeSetInternalLogProvider( TestDatabaseManagementServiceBuilder factory )
    {
        return ( internalLogProvider == null ) ? factory : factory.setInternalLogProvider( internalLogProvider );
    }
}
