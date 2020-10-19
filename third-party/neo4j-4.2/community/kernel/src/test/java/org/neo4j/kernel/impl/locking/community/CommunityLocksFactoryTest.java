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
package org.neo4j.kernel.impl.locking.community;

import org.junit.jupiter.api.Test;

import org.neo4j.configuration.Config;
import org.neo4j.kernel.impl.locking.Locks;
import org.neo4j.lock.ResourceTypes;
import org.neo4j.time.Clocks;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertNotSame;

class CommunityLocksFactoryTest
{

    @Test
    void createDifferentCommunityLockManagers()
    {
        CommunityLocksFactory factory = new CommunityLocksFactory();
        Locks locks1 = factory.newInstance( Config.defaults(), Clocks.nanoClock(), ResourceTypes.values() );
        Locks locks2 = factory.newInstance( Config.defaults(), Clocks.nanoClock(), ResourceTypes.values() );
        assertNotSame( locks1, locks2 );
        assertThat( locks1 ).isInstanceOf( CommunityLockManger.class );
        assertThat( locks2 ).isInstanceOf( CommunityLockManger.class );
    }
}
