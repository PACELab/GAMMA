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

import org.junit.jupiter.api.Test;

import java.util.concurrent.ThreadLocalRandom;

import org.neo4j.kernel.impl.store.record.RelationshipRecord;

import static org.assertj.core.api.Assertions.assertThat;

class QueueDistributionTest
{

    private static final int MAX_NUMBER_OF_THREADS = 1_000_000;
    private static final int NUMBER_OF_DISTRIBUTION_ITERATIONS = 1000;

    @Test
    void roundRobinRecordDistribution() throws Exception
    {
        testRecordDistribution( QueueDistribution.ROUND_ROBIN );
    }

    @Test
    void relationshipNodesDistribution() throws InterruptedException
    {
        testRecordDistribution( QueueDistribution.RELATIONSHIPS );
    }

    private static void testRecordDistribution( QueueDistribution queueDistribution ) throws InterruptedException
    {
        ThreadLocalRandom randomGenerator = ThreadLocalRandom.current();
        int numberOfThreads = randomGenerator.nextInt( MAX_NUMBER_OF_THREADS );
        int recordsPerCpu = randomGenerator.nextInt( Integer.MAX_VALUE );
        QueueDistribution.QueueDistributor<RelationshipRecord> distributor =
                queueDistribution.distributor( recordsPerCpu, numberOfThreads );
        for ( int iteration = 0; iteration <= NUMBER_OF_DISTRIBUTION_ITERATIONS; iteration++ )
        {
            RelationshipRecord relationshipRecord = new RelationshipRecord( 1 );
            relationshipRecord.setFirstNode( nextLong( randomGenerator ) );
            relationshipRecord.setSecondNode( nextLong( randomGenerator ) );
            distributor.distribute( relationshipRecord, ( record, qIndex ) -> assertThat( qIndex ).as(
                    "Distribution index for record " + record + " should be within a range of available executors, while expected records per cpu is: " +
                            recordsPerCpu ).isGreaterThanOrEqualTo( 0 ).isLessThan( numberOfThreads ) );
        }
    }

    private static long nextLong( ThreadLocalRandom randomGenerator )
    {
        return randomGenerator.nextLong();
    }
}
