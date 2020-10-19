FROM strimzi/kafka:latest-kafka-2.4.0

ARG neo4j_streams_release=3.5.6
ARG kafka_connect_neo4j_version=1.0.6
ARG kafka_connect_jdbc_version=5.4.1

USER root:root

WORKDIR /opt/kafka/plugins/

RUN set -eux; \
    mkdir -p /opt/confluent-hub/; \
    curl -SsL http://client.hub.confluent.io/confluent-hub-client-latest.tar.gz \
        | tar --extract --gzip --directory="/opt/confluent-hub/" \
    ; \
    /opt/confluent-hub/bin/confluent-hub install --no-prompt confluentinc/kafka-connect-jdbc:${kafka_connect_jdbc_version}; \
    mv /opt/confluent-hub/share/confluent-hub-components/* $(pwd); \
    rm -rf /opt/confluent-hub/;

# Special installation Process for Neo4j sink because confluent hub does not have the last tag
RUN set -eux; \
    yum install -y unzip; \
    curl -SsL \
       https://github.com/neo4j-contrib/neo4j-streams/releases/download/${neo4j_streams_release}/neo4j-kafka-connect-neo4j-${kafka_connect_neo4j_version}.zip \
       > neo4j-kafka-connect-neo4j.zip; \
    unzip neo4j-kafka-connect-neo4j.zip; \
    rm -rf neo4j-kafka-connect-neo4j.zip; \
    yum remove -y unzip; \
    yum clean all;

WORKDIR /opt/kafka/

USER 1001