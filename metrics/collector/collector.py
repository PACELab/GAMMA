"""
A python3 stats collector for gathering cAdvisor metrics and storing them in Redis.

cAdvisor API format:
https://github.com/google/cadvisor/blob/master/info/v1/container.go

TODO: Migrate to v2
https://github.com/google/cadvisor/blob/master/info/v2/container.go

Running w/gunicorn:

gunicorn --workers=1 --log-level debug --log-file=- --bind 0.0.0.0:$COLLECTOR_PORT 'collector:build_app()'
"""

from wsgiref import simple_server
from multiprocessing import Process

import os
import falcon
import redis
import time
import json
import logging

REDIS_HOST=os.getenv('COLLECTOR_REDIS_HOST', 'redis.local')
REDIS_PORT=int(os.getenv('COLLECTOR_REDIS_PORT', '6379'))
PORT=int(os.getenv('COLLECTOR_PORT', '8787'))
STATS_LEN=int(os.getenv('STATS_LEN', '1440'))

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

class PurgeHandler():
    def __init__(self, redis_host, redis_port):
        self.running = True
        self.logger = logging.getLogger(__name__)
        self.redis_host = redis_host
        self.redis_port = redis_port

    def kill(self):
        self.running = False

    def process(self):
        while self.running:
            r = redis.StrictRedis(host=self.redis_host, port=self.redis_port)
            names = r.smembers('names')
            for name in names:
                # Check for expired names
                name = name.decode()
                if not r.exists('name:%s' % name):
                    # Clean up anything that hasn't sent data lately
                    self.logger.debug('Cleaning up %s.' % name)
                    r.srem('names', name)
                    r.delete('stats:%s' % name)

            time.sleep(60)

class StatHandler():
    def __init__(self, redis_host, redis_port, metadata_fun=None):
        """
        Create the handler. Use the optional metadata_fun to gather external data to store with the stats.

        :param redis_host: Where to connect to Redis
        :param redis_port: Port Redis is listening on
        :param metadata_fun: A function for grabbing external metadata (see _get_metadata_noop)
        """
        self.logger = logging.getLogger(__name__)
        self.redis_host = redis_host
        self.redis_port = redis_port
        if metadata_fun:
            self._get_metadata = metadata_fun
        else:
            self._get_metadata = self._get_metadata_default

    def process(self, entry, remote_ip):
        """
        Given a stats entry and IP, store the stats and machine data in Redis.
        """
        self.logger.debug(entry)
        # print(entry)

        r = redis.StrictRedis(host=self.redis_host, port=self.redis_port)
        ts = entry['timestamp']

        # Take the machine info and store it by IP
        machine = entry['machine'] 
        r.set('ip:%s' % remote_ip, json.dumps(machine)) 
        r.expire('ip:%s' % remote_ip, 1*24*60*60)

        # For each container being reported, record info in Redis
        for stat in entry['stats']:
            name = stat['name']  # Grab the container name

            # Keep metadata about the container (for now just the IP it is on)
            # If a container stops sending data it will eventually be removed
            container_data = {'remote_ip': remote_ip}
            r.set('name:%s' % name, json.dumps(container_data))
            r.expire('name:%s' % name, 1*24*60*60) # See PurgeHandler above for how expired containers are handled

            stat['ts'] = ts  # Add the timestamp to each entry

            # Check to see if this is a new container (i.e. not in the set)
            if not r.sismember('names', name):

                # Lookup external data that might be of use upstream
                if not self._get_metadata(name, r, ignore_fail=False):
                    # Skip this entry if we have trouble looking up info
                    continue

                # Record the name of this container
                r.sadd('names', name)

            # An assumption is made here that all container names are unique across all machines.
            # Adding machine info to each key would allow for duplicate names (for example using the IP).
            # Also assumes that data is coming in time order per container.
            # These are not ideal assumptions but they are fine for our purposes and can be improved upon in the future.
            r.lpush('stats:%s' % name, json.dumps(stat))

            # Trim the list of entries for this container to bound the amount of data being stored
            r.ltrim('stats:%s' % name, 0, STATS_LEN) # Store ~1 day's worth of data per container

    def _get_metadata_default(self, name, r, ignore_fail=False):
        """
        Handles collection of ancillary metadata. The default implementation is a no op.

        At Catalyze we use a function here that pulls data from an internal API
        to find related details about the container.

        This function can be adjusted to suit individual needs, although a function can be passed
        into the constructor so long as it has the same parameters as this one.

        :param name: The container name
        :param r: The redis connection
        :param ignore_fail: boolean indicating if we should ignore failures in lookup or not
        :return: True if ignore_fail is True or if the data was looked up successfully, else returns False
        """
        return True

class CadvisorMetricsResource(object):
    def __init__(self, redis_host, redis_port, metadata_fun=None):
        self.logger = logging.getLogger(__name__)
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.metadata_fun = metadata_fun
        self.fmt = lambda obj: json.dumps(obj, indent=4, sort_keys=True) # Handy JSON function

    def on_post(self, req, resp):
        """
        Receives aggregated cAdvisor stats from docker hosts.
        Send them on to a spawned StatHandler to avoid blocking the sender.

        :param req: The HTTP request
        :param resp: The HTTP response
        """
        # Get the IP of the host sending stats
        remote_ip = req.env['REMOTE_ADDR']

        # The cadvisor data is JSON in the request body
        body = req.stream.read()
        if not body:
            self.logger.error('Empty body provided when returning a command result.')
            raise falcon.HTTPBadRequest('Empty request body', 'A valid JSON document is required.')

        entry = json.loads(body.decode())

        # Set up to process the stats in the background so as not to block.
        handler = StatHandler(self.redis_host, self.redis_port, self.metadata_fun)
        p = Process(target=handler.process, args=(entry, remote_ip,))
        p.start()

        resp.set_header('Content-Type', 'application/json')
        resp.status = falcon.HTTP_200
        resp.body = self.fmt({})

class CollectorApp():
    def __init__(self):
        self.r = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT)

    def build_app(self, metadata_fun=None):
        # Launch background process to clean up stale stats/containers
        ph = PurgeHandler(REDIS_HOST, REDIS_PORT)
        p = Process(target=ph.process, args=())
        p.start()

        app = falcon.API()
        resource = CadvisorMetricsResource(REDIS_HOST, REDIS_PORT, metadata_fun)
        app.add_route('/cadvisor/metrics/{env_id}', resource)
        app.add_route('/cadvisor/metrics', resource)
        return app

    def get_stat(self, id):
        stat = self.r.get(id)
        return stat

def build_app():
    c = CollectorApp()
    return c.build_app()

if __name__ == '__main__':
    # For testing outside a WSGI like gunicorn
    collector = CollectorApp()
    httpd = simple_server.make_server('0.0.0.0', PORT, collector.build_app())
    httpd.serve_forever()
