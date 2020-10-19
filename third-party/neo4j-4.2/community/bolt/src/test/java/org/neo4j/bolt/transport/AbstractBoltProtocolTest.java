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
package org.neo4j.bolt.transport;

import io.netty.channel.ChannelHandler;
import io.netty.channel.embedded.EmbeddedChannel;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.Iterator;
import java.util.Map;

import org.neo4j.bolt.BoltChannel;
import org.neo4j.bolt.BoltProtocol;
import org.neo4j.bolt.BoltProtocolVersion;
import org.neo4j.bolt.messaging.BoltRequestMessageReader;
import org.neo4j.bolt.messaging.BoltResponseMessageWriter;
import org.neo4j.bolt.packstream.Neo4jPack;
import org.neo4j.bolt.runtime.BoltConnection;
import org.neo4j.bolt.runtime.BoltConnectionFactory;
import org.neo4j.bolt.runtime.BookmarksParser;
import org.neo4j.bolt.runtime.statemachine.BoltStateMachineFactory;
import org.neo4j.bolt.transport.pipeline.ChunkDecoder;
import org.neo4j.bolt.transport.pipeline.HouseKeeper;
import org.neo4j.bolt.transport.pipeline.MessageAccumulator;
import org.neo4j.bolt.transport.pipeline.MessageDecoder;
import org.neo4j.logging.internal.LogService;
import org.neo4j.logging.internal.NullLogService;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.neo4j.bolt.testing.BoltTestUtil.newTestBoltChannel;

class AbstractBoltProtocolTest
{
    private final EmbeddedChannel channel = new EmbeddedChannel();

    @AfterEach
    void cleanup()
    {
        channel.finishAndReleaseAll();
    }

    @Test
    void shouldInstallChannelHandlersInCorrectOrder() throws Throwable
    {
        // Given
        BoltChannel boltChannel = newTestBoltChannel( channel );
        BoltConnectionFactory connectionFactory = mock( BoltConnectionFactory.class );
        when( connectionFactory.newConnection( eq( boltChannel ), any(), any() ) ).thenReturn( mock( BoltConnection.class ) );
        BoltProtocol boltProtocol =
                new TestAbstractBoltProtocol( boltChannel, connectionFactory, mock( BoltStateMachineFactory.class ),
                        NullLogService.getInstance(), mock( TransportThrottleGroup.class ) );

        // When
        boltProtocol.install();

        Iterator<Map.Entry<String,ChannelHandler>> handlers = channel.pipeline().iterator();
        assertThat( handlers.next().getValue() ).isInstanceOf( ChunkDecoder.class );
        assertThat( handlers.next().getValue() ).isInstanceOf( MessageAccumulator.class );
        assertThat( handlers.next().getValue() ).isInstanceOf( MessageDecoder.class );
        assertThat( handlers.next().getValue() ).isInstanceOf( HouseKeeper.class );

        assertFalse( handlers.hasNext() );
    }

    private static class TestAbstractBoltProtocol extends AbstractBoltProtocol
    {
        private static final BoltProtocolVersion DUMMY_VERSION = new BoltProtocolVersion( 0, 0 );

        TestAbstractBoltProtocol( BoltChannel channel, BoltConnectionFactory connectionFactory, BoltStateMachineFactory stateMachineFactory,
                                  LogService logging, TransportThrottleGroup throttleGroup )
        {
            super( channel, connectionFactory, stateMachineFactory, logging, throttleGroup );
        }

        @Override
        protected Neo4jPack createPack()
        {
            return mock( Neo4jPack.class );
        }

        @Override
        protected BoltRequestMessageReader createMessageReader( BoltConnection connection,
                BoltResponseMessageWriter messageWriter, BookmarksParser bookmarksParser, LogService logging )
        {
            return mock( BoltRequestMessageReader.class );
        }

        @Override
        protected BoltResponseMessageWriter createMessageWriter( Neo4jPack neo4jPack, LogService logging )
        {
            return mock( BoltResponseMessageWriter.class );
        }

        @Override
        public BoltProtocolVersion version()
        {
            return DUMMY_VERSION;
        }
    }
}
