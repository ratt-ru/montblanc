import argparse
import fcntl
import json
import logging
import os
import select
import socket
import struct

class SyncError(Exception):
    pass

PING = 'PING'
PONG = 'PONG'

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--interface', default='eth0')
parser.add_argument('-p', '--port', default=8888)
parser.add_argument('--no-start', dest='start', action='store_false')
parser.add_argument('--start', dest='start', action='store_true')
parser.set_defaults(start=False)
args = parser.parse_args()

def get_ip_address(ifname):
    """ Hack to get IP address from the interface """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', ifname[:15])
    )[20:24])

# Track client connections
connections = {}
lost = set()

# Determine host address to bind a server socket to
host_address = (get_ip_address(args.interface), args.port)

try:
    logging.info('Server listening on {a}'.format(a=host_address))

    host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host_socket.bind(host_address)
    host_socket.listen(5)

    while True:
        client_socket, client_address = host_socket.accept()
        logging.info('Connection from {a}'.format(a=client_address))
        connections[client_socket] = (client_socket, client_address)

except KeyboardInterrupt:
    logging.info('Ctrl^C received')

    logging.info('Pinging {n} connection(s)'.format(n=len(connections)))

    for k, (cs, ca) in connections.iteritems():
        try:
            cs.send(PING)
        except socket.error as e:
            logging.warn('Lost connection to {a}'.format(a=ca))
            lost.add((cs,ca))

    for k, (cs, ca) in connections.iteritems():
        try:
            if cs.recv(len(PONG)) != PONG:
                raise SyncError()
        except (socket.error, SyncError) as e:
            logging.warn('Lost connection to {a}'.format(a=ca))
            lost.add((cs, ca))

    logging.info('Lost {n} connection(s)'.format(n=len(lost)))

    connections = { k : c for k, c in connections.iteritems()
        if c not in lost }

    logging.info('Creating cluster specification for {n} workers'.format(
        n=len(connections)))
    # Create the lists of workers and master urls
    master_list = ['{ip}:{port}'.format(ip=host_address[0], port=host_address[1])]
    worker_list = ['{ip}:{port}'.format(ip=ip, port=port) for (ip, port) in
        (s.getpeername() for s, _ in connections.itervalues())]

    logging.info('Master node(s) {n}'.format(n=master_list))
    logging.info('Worker node(s) {n}'.format(n=worker_list))

    cluster = { 'worker' : worker_list, 'master' : master_list }

    # Transmit cluster specification to connected clients
    for i, (cs, ca) in enumerate(connections.itervalues()):
        data = { 'cluster' : cluster, 'job' : 'worker', 'task' : i }

        logging.info('Sending specification to {ca}'.format(ca=ca))
        cs.send(json.dumps(data))

finally:
    # Close client sockets
    for cs, address in connections.itervalues():
        logging.info('Closing connection to {c}'.format(c=address))
        cs.shutdown(socket.SHUT_RDWR)
        cs.close()

    for cs, address in lost:
        logging.info('Closing connection to {c}'.format(c=address))
        cs.close()

    # Close server socket
    host_socket.shutdown(socket.SHUT_RDWR)
    host_socket.close()
    logging.info('Closing host socket {h}'.format(h=host_address))

logging.info("Cluster specification\n{c}".format(c=cluster))

if args.start is True:
    import tensorflow as tf
    import montblanc
    import time

    server = tf.train.Server(cluster, job_name='master', task_index=0)
    logging.info("Server Target is '{st}'".format(st=server.target))
    logging.info("Server Job Name is '{j}'".format(j=server.server_def.job_name))
    logging.info("Server Task Index is '{t}'".format(t=server.server_def.task_index))

    slvr_cfg = montblanc.rime_solver_cfg(
        mem_budget=1024*1024*1024,
        data_source='default',
        dtype='double',
        auto_correlations=False,
        version='tf',
        tf_server_target=server.target,
        tf_job_name=server.server_def.job_name,
        tf_task_index=server.server_def.task_index,
        tf_cluster=cluster)

    slvr = montblanc.rime_solver(slvr_cfg)

    logging.info("Created tensorflow solver")

    slvr.solve()

    time.sleep(10)