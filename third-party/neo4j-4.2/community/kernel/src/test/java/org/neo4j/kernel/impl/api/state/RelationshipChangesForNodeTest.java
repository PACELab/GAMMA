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
package org.neo4j.kernel.impl.api.state;

import org.eclipse.collections.api.iterator.LongIterator;
import org.junit.jupiter.api.Test;

import org.neo4j.graphdb.Direction;

import org.neo4j.memory.EmptyMemoryTracker;

import static org.assertj.core.api.Assertions.assertThat;
import static org.neo4j.collection.PrimitiveLongCollections.asArray;
import static org.neo4j.storageengine.api.RelationshipDirection.INCOMING;
import static org.neo4j.storageengine.api.RelationshipDirection.LOOP;
import static org.neo4j.storageengine.api.RelationshipDirection.OUTGOING;

class RelationshipChangesForNodeTest
{

    @Test
    void shouldGetRelationships()
    {
        RelationshipChangesForNode changes = RelationshipChangesForNode.createRelationshipChangesForNode(
                RelationshipChangesForNode.DiffStrategy.ADD, EmptyMemoryTracker.INSTANCE );

        final int TYPE = 2;

        changes.addRelationship( 1, TYPE, INCOMING );
        changes.addRelationship( 2, TYPE, OUTGOING );
        changes.addRelationship( 3, TYPE, OUTGOING );
        changes.addRelationship( 4, TYPE, LOOP );
        changes.addRelationship( 5, TYPE, LOOP );
        changes.addRelationship( 6, TYPE, LOOP );

        LongIterator rawRelationships = changes.getRelationships();
        assertThat( asArray( rawRelationships ) ).containsExactly( 1, 2, 3, 4, 5, 6 );
    }

    @Test
    void shouldGetRelationshipsByTypeAndDirection()
    {
        RelationshipChangesForNode changes = RelationshipChangesForNode.createRelationshipChangesForNode(
                RelationshipChangesForNode.DiffStrategy.ADD, EmptyMemoryTracker.INSTANCE );

        final int TYPE = 2;
        final int DECOY_TYPE = 666;

        changes.addRelationship( 1, TYPE, INCOMING );
        changes.addRelationship( 2, TYPE, OUTGOING );
        changes.addRelationship( 3, TYPE, OUTGOING );
        changes.addRelationship( 4, TYPE, LOOP );
        changes.addRelationship( 5, TYPE, LOOP );
        changes.addRelationship( 6, TYPE, LOOP );

        changes.addRelationship( 10, DECOY_TYPE, INCOMING );
        changes.addRelationship( 11, DECOY_TYPE, OUTGOING );
        changes.addRelationship( 12, DECOY_TYPE, LOOP );
        LongIterator rawIncoming = changes.getRelationships( Direction.INCOMING, TYPE );
        assertThat( asArray( rawIncoming ) ).containsExactly( 1, 4, 5, 6 );

        LongIterator rawOutgoing = changes.getRelationships( Direction.OUTGOING, TYPE );
        assertThat( asArray( rawOutgoing ) ).containsExactly( 2, 3, 4, 5, 6 );
    }
}
