"""
Require: Microservice execution history graph G
Attributes: childNodes, lastReturnedChild
1: procedure LONGESTPATH(G, currentNode)
2: path ← ∅
3: path.add(currentNode)
4: if currentNode.childNodes == None then
5: Return path
6: end if
7: lrc ← currentNode.lastReturnedChild
8: path.extend(LONGESTPATH(G, lrc))
9: for each cn in currentNode.childNodes do
10: if cn.happensBefore(lrc) then
11: path.extend(LONGESTPATH(G, cn))
12: end if
13: end for
14: Return path
15: end procedure

Require: Critical Path CP, Request Latencies T
1: procedure CRITICALCOMPONENT(G, T)
2: candidates ← ∅
3: TCP ← T.getTotalLatency() . Vector of CP latencies
4: for i ∈ CP do
5: Ti ← T.getLatency(i)
6: T99 ← Ti.percentile(99)
7: T50 ← Ti.percentile(50)
8: RI ← PCC(Ti,TCP) . Relative Importance
9: CI ← T99/T50 . Congestion Intensity
10: if SVM.classi f y(RI,CI) == True then
11: candidates.append(i)
12: end if
13: end for
14: Return candidates
15: end procedure
"""

from neo4j import GraphDatabase
import logging
import minibatch_svm

DATABASE = 'localhost:7687'
USERNAME = 'neo4j'
PASSWORD = 'password'
DURATION = 5 # s
ROOT_ID = 'FFFFFFFFFFFFFFFF'

class TracingPipeline():
    def __init__(self):
        # BoltDriver with no encryption
        self.driver = GraphDatabase.driver('bolt://'+DATABASE, auth=(USERNAME, PASSWORD)) # thread-safe
        # self.driver = GraphDatabase.driver('neo4j://'+DATABASE, auth=(USERNAME, PASSWORD))

    def read_tracing_data(self):
        with self.driver.session() as session:
            data = session.read_transaction(get_tracing_data, DURATION)
            return data

    def get_tracing_data(tx, interval):
        data = {}
        curr_time = int(round(time.time() * 1000000))
        from_time = curr_time - interval * 1000000
        # result = tx.run('MATCH ((n:Node) WHERE n.timestamp > ' + str(from_time) + ' AND n.timestamp < ' + str(curr_time)) + ')-[:CHILD_OF]->(root:Nde {id:' + ROOT_ID + '})'
        result = tx.run('MATCH (child:Node)-[:CHILD_OF]->(parent:Node) ' +\
                        'WHERE id(parent) = {' + ROOT_ID + '} AND parent.timestamp > ' + str(from_time) + ' AND parent.timestamp < ' + str(curr_time)) +\
                        'WITH child' +\
                        'MATCH childPath=(child)-[:CHILD*0..]->(endNode:Node)' +\
                        'with childPath, endNode order by endNode.id' +\
                        'with collect(childPath) as paths' + \
                        'CALL apoc.convert.toTree(paths) yield value' +\
                        'RETURN value'
        # for record in result:
        #     pass
        return result

    def write_candidates(self):
        with self.driver.session() as session:
            id = session.write_transaction(put_candidates, candidates)
            return id

    def put_candidates(tx, candidate):
        result = tx.run('CREATE (n:' + candidate.id + ' {slo_retainment: ' + str(candidate.slo_retainment) + ', curr_arrival_rate: ' + str(candidate.curr_arrival_rate)+ ', rate_ratio: ' + str(candidate.rate_ratio) + ', percentages' + str(candidate.percentages) + '}) RETURN id(n) AS node_id')
        record = result.single()
        return record['node_id']

    def run(self):
        while True:
            # get all traces in last DURATION
            traces = self.read_tracing_data()

            for trace in traces:
                # get all spans in a trace
                spans = trace['spans']
                critical_path = []
                max_span = spans[0]
                max_endtime = spans[0]['start_time'] + spans[0]['duration']
                for span in spans:
                    endtime =span['start_time'] + span['duration']
                    if endtime > max_endtime:
                        max_endtime = endtime
                        max_span = span

                # construct the critical path
                critical_path.append(max_span)
                child_nodes = max_span['child_nodes']
                while child_nodes != None:
                    last = child_nodes[len(child_nodes)-1]
                    critical_path.append(last)
                    child_nodes = last['child_nodes']

                # get the critical candidates
                candidates = minibatch_svm.inference(critical_path)
                self.write_candidates(candidates)

    def close(self):
        self.driver.close()

if __name__ == '__main__':
    logging.basicConfig()
    pipeline = TracingPipeline()
    pipeline.run()
