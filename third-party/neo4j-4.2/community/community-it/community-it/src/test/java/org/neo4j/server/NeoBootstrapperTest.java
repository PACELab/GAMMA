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
package org.neo4j.server;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.parallel.ResourceLock;
import org.junit.jupiter.api.parallel.Resources;

import java.nio.file.Files;
import java.nio.file.Path;

import org.neo4j.internal.helpers.collection.MapUtil;
import org.neo4j.test.extension.Inject;
import org.neo4j.test.extension.SuppressOutputExtension;
import org.neo4j.test.extension.testdirectory.TestDirectoryExtension;
import org.neo4j.test.rule.SuppressOutput;
import org.neo4j.test.rule.TestDirectory;

import static org.assertj.core.api.Assertions.assertThat;

@TestDirectoryExtension
@ExtendWith( SuppressOutputExtension.class )
@ResourceLock( Resources.SYSTEM_OUT )
class NeoBootstrapperTest
{
    @Inject
    private TestDirectory homeDir;
    @Inject
    private SuppressOutput suppress;
    private NeoBootstrapper neoBootstrapper;

    @AfterEach
    void tearDown()
    {
        if ( neoBootstrapper != null )
        {
            neoBootstrapper.stop();
        }
    }

    @Test
    void shouldNotThrowNullPointerExceptionIfConfigurationValidationFails() throws Exception
    {
        // given
        neoBootstrapper = new CommunityBootstrapper();

        Path dir = Files.createTempDirectory( "test-server-bootstrapper" );

        // when
        neoBootstrapper.start( dir, MapUtil.stringMap() );

        // then no exceptions are thrown and
        assertThat( suppress.getOutputVoice().lines() ).isNotEmpty();
    }
}
