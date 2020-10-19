from __future__ import print_function
"""
Rolls up stats by IP address.
"""

import redis
import json
import sys

HOST='localhost'
PORT=6379

# Adjust the below value to change the time range, for example use 60 to get the last min's stats.
NUM_ENTRIES=30 # Number of entries to aggregate. 1 entry per sec
# NUM_ENTRIES=1440

def error(msg):
    print(msg, file=sys.stderr)

def aggregate_stats(containers):
    """
    Given a list of containers, fetch all the stats and aggregate them per-time interval.

    :param containers: List of container names to aggregate
    :return: A list of stat entry dicts, one per time interval
    """
    result = []
    r = redis.StrictRedis(host=HOST, port=PORT)
    for container in containers:
        # Load up the latest stats for this container. The newest entries are the front of the list
        stats = r.lrange('stats:%s' % container, 0, NUM_ENTRIES)
        for i, stat in enumerate(stats):
            stat = json.loads(stat)
            if i == len(result):
                # This is the first entry for this time interval, use it as is after removing name
                del stat['name']
                result.append(stat)
            else:
                # Increment values for the current time interval using this container's stats
                result[i]['cpu']['usage'] += stat['cpu']['usage']
                result[i]['cpu']['load']['min'] += stat['cpu']['load']['min']
                result[i]['cpu']['load']['ave'] += stat['cpu']['load']['ave']
                result[i]['cpu']['load']['max'] += stat['cpu']['load']['max']
                result[i]['memory']['min'] += stat['memory']['min']
                result[i]['memory']['ave'] += stat['memory']['ave']
                result[i]['memory']['max'] += stat['memory']['max']
                result[i]['network']['tx_bytes']['min'] += stat['network']['rx_bytes']['min']
                result[i]['network']['tx_bytes']['ave'] += stat['network']['rx_bytes']['ave']
                result[i]['network']['tx_bytes']['max'] += stat['network']['rx_bytes']['max']
                result[i]['network']['rx_bytes']['min'] += stat['network']['rx_bytes']['min']
                result[i]['network']['rx_bytes']['ave'] += stat['network']['rx_bytes']['ave']
                result[i]['network']['rx_bytes']['max'] += stat['network']['rx_bytes']['max']
                result[i]['network']['tx_packets']['min'] += stat['network']['rx_packets']['min']
                result[i]['network']['tx_packets']['ave'] += stat['network']['rx_packets']['ave']
                result[i]['network']['tx_packets']['max'] += stat['network']['rx_packets']['max']
                result[i]['network']['rx_packets']['min'] += stat['network']['rx_packets']['min']
                result[i]['network']['rx_packets']['ave'] += stat['network']['rx_packets']['ave']
                result[i]['network']['rx_packets']['max'] += stat['network']['rx_packets']['max']
                result[i]['diskio']['async'] += stat['diskio']['async']
                result[i]['diskio']['sync'] += stat['diskio']['sync']
                result[i]['diskio']['write'] += stat['diskio']['write']
                result[i]['diskio']['read'] += stat['diskio']['read']

    return result

r = redis.StrictRedis(host=HOST, port=PORT)

names = r.smembers('names') # Get the list of container names
by_ip = {}

# For each container, determine the IP and group containers by IP
for name in names:
    name_info = r.get('name:%s' % name)
    if not name_info:
         print('Missing data for name: %s' % name)
         continue
 
    name_data = json.loads(r.get('name:%s' % name).decode())
    try:
        ip = name_data.get('remote_ip', None)
    except:
        r.delete('name:%s' % name)
    if not ip:
        print('Missing IP address for name: %s' % name)
        continue

    entry = by_ip.get(ip, set())
    entry.add(name)
    by_ip[ip] = entry

# Compute the final result, aggregating all the container stats for each IP
print('interval,ip,memory,cpu,txbytes,rxbytes,io_async,io_sync,io_read,io_write')
for ip, containers in by_ip.iteritems():
    aggregate = aggregate_stats(containers)
    #result = {'ip': ip, 'stats': aggregate}
    for i, stat in enumerate(aggregate):
        line = '%d,%s,%d,%d,%d,%d,%d,%d,%d,%d' % (i, ip, stat['memory']['ave'], stat['cpu']['usage'], stat['network']['tx_bytes']['ave'], stat['network']['rx_bytes']['ave'], stat['diskio']['async'],stat['diskio']['sync'], stat['diskio']['read']/1024, stat['diskio']['write']/1024)
        print(line)
