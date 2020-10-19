from neo4j import GraphDatabase

from wsgiref import simple_server
from concurrent import futures
import time
import logging
import os
import threading

import grpc
import message_pb2
import message_pb2_grpc

from metrics.collector import collector

import actions

REDIS_HOST=os.getenv('COLLECTOR_REDIS_HOST', '192.168.222.5')
REDIS_PORT=int(os.getenv('COLLECTOR_REDIS_PORT', '6379'))
PORT=int(os.getenv('COLLECTOR_PORT', '8787'))
STATS_LEN=int(os.getenv('STATS_LEN', '1440'))

NUM_WORKERS = 32
# METRIC_SERVER = '10.2.2.0'
# CADVISOR_API = '10.2.2.1'
DATABASE = 'localhost:7687'
USERNAME = 'neo4j'
PASSWORD = 'GorgeousPassword'

PATH_TO_CONTAINER_INFO = 'pods.csv'

def register_containers():
    f = open(PATH_TO_CONTAINER_INFO, 'r')
    lines = f.readlines()

    db = {}
    for line in lines:
        l = line.strip().split(',')
        db[l[0]] = l[1]
    
    return db

def init_collector(collector_app):
    httpd = simple_server.make_server('0.0.0.0', PORT, collector_app)
    # TODO - Add stopping statements
    httpd.serve_forever()

class InteractionServicer(message_pb2_grpc.InteractionServicer):
    """Provides methods that implement functionality of route guide server."""
    def __init__(self):
        # BoltDriver with no encryption
        self.driver = GraphDatabase.driver('bolt://'+DATABASE, auth=(USERNAME, PASSWORD)) # thread-safe
        # self.driver = GraphDatabase.driver('neo4j://'+DATABASE, auth=(USERNAME, PASSWORD))
        self.container_map = register_containers()
        self.collector = collector.CollectorApp()
        try:
            threading.Thread(target=init_collector, args=(self.collector.build_app())).start()
        except:
            print("Error: unable to start thread!")

    # request: ComponentId
    # response: ToClientMessage
    def GetState(self, request, context):
        metrics_stat = self.collector.get_stat(request.id)
        tracing_stat = self.read_tracing_stat(request.id)
        message = message_pb2.ToClientMessage()
        message.name = request.name
        message.node = request.node
        message.id = request.id
        message.usage.cpu = metrics_stat['cpu']
        message.usage.memory = metrics_stat['memory']
        message.usage.llc = metrics_stat['cache']
        message.usage.network = metrics_stat['network']
        message.usage.io = metrics_stat['diskio']
        message.limit = None;
        message.other['slo_retainment'] = tracing_stat['slo_retainment'];
        message.other['curr_arrival_rate'] = tracing_stat['curr_arrival_rate'];
        message.other['rate_ratio'] = tracing_stat['rate_ratio'];
        message.other['percentages'] = tracing_stat['percentages'];
        message.status = 'OK';
        return message
    
    # request: ToServerMessage
    # response: ToClientMessage
    def PerformAction(self, request, context):
        if request.id in self.container_map:
            # execute action
            actions.cpu(request.id, request.action.cpu, self.container_map[request.id].cores)
            actions.memory(request.id, request.action.memory, self.container_map[request.id].cores)
            actions.llc(request.id, request.action.llc)
            actions.blkio(request.id, request.action.io)
            actions.network(request.id, request.action.network)
            # response
            metrics_stat = self.collector.get_stat(request.id)
            tracing_stat = self.read_tracing_stat(request.id)
            message = message_pb2.ToClientMessage()
            message.name = request.name
            message.id = request.id
            message.usage.cpu = metrics_stat['cpu']
            message.usage.memory = metrics_stat['memory']
            message.usage.llc = metrics_stat['cache']
            message.usage.network = metrics_stat['network']
            message.usage.io = metrics_stat['diskio']
            message.limit = None;
            message.other['slo_retainment'] = tracing_stat['slo_retainment'];
            message.other['curr_arrival_rate'] = tracing_stat['curr_arrival_rate'];
            message.other['rate_ratio'] = tracing_stat['rate_ratio'];
            message.other['percentages'] = tracing_stat['percentages'];
            message.status = 'OK';
            return message

    def read_tracing_stat(id):
        with self.driver.session() as session:
            stat = session.read_transaction(get_stat_of, id)
            return stat

    def get_stat_of(tx, id):
        stat = {}
        curr_time = int(round(time.time() * 1000000))
        from_time = curr_time - 300000000
        if id == 'DEFAULT':
            result = tx.run('MATCH (n) WHERE n.timestamp > ' + str(from_time) + ' AND n.timestamp < ' + str(curr_time)) + ' RETURN n'
        else:
            result = tx.run('MATCH (n:' + id + ') WHERE n.timestamp > ' + str(from_time) + ' AND n.timestamp < ' + str(curr_time)) + ' RETURN n'
        # for record in result:
        #     pass
        stat['slo_retainment'] = result[0]['slo_retainment']
        stat['curr_arrival_rate'] = result[0]['arrival_rate']
        stat['rate_ratio'] = result[0]['rate_ratio']
        stat['percentages'] = result[0]['percentages']

    def close(self):
        self.driver.close()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=NUM_WORKERS))
    message_pb2_grpc.add_InteractionServicer_to_server(
        InteractionServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()
