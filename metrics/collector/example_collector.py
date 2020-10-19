"""
An example of making a custom collector. Each time a sender reports data, the collector
will call _get_metadata_example() to allow for external information to be gathered about
the container. An example would be to lookup container information in a database and
then store it in Redis along with the metrics information to add more context to a container.
"""

from wsgiref import simple_server

import os
import logging
import collector

PORT=int(os.getenv('COLLECTOR_PORT', '8787'))

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

def _get_metadata_example(self, name, r, ignore_fail=False):
    """
    Handles collection of data related to the container from some external source (database lookup, etc).
    :param name: The container name
    :param r: The redis connection
    :param ignore_fail: boolean indicating if we should ignore failures in lookup or not
    :return: True if ignore_fail is True or if the data was looked up successfully, else returns False
    """

    # Logic goes here...
    # Implement as you see fit.

    return True

def build_app():
    c = collector.CollectorApp()
    return c.build_app(_get_metadata_example)

if __name__ == '__main__':
    # For testing outside a WSGI like gunicorn
    c = collector.CollectorApp()
    httpd = simple_server.make_server('0.0.0.0', PORT, c.build_app(_get_metadata_example))
    httpd.serve_forever()
