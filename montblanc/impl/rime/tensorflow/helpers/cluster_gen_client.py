import argparse
import json
import logging
import os
import socket

CHUNK_SIZE = 1024
DEFAULT_PORT = 8888
PING = 'PING'
PONG = 'PONG'

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

# Command line parser
parser = argparse.ArgumentParser()
parser.add_argument('server', type=str)
parser.add_argument('-c', '--clean', default=True, type=bool)
parser.add_argument('--no-start', dest='start', action='store_false')
parser.add_argument('--start', dest='start', action='store_true')
parser.set_defaults(start=False)
args = parser.parse_args()

port_index = args.server.rfind(':')

if port_index != -1:
    port = int(args.server[port_index+1:])
    address = (args.server[:port_index], port)
else:
    address = (args.server, DEFAULT_PORT)

chunks = []

try:
    logging.info("Connecting to {a}".format(a=address))
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(address)

    # Wait for the server to signal specification is being sent
    ping = s.recv(len(PING))

    if ping != PING:
        raise ValueError("Expected {p}, received {i}".format(p=PING, i=ping))

    # Signal server we're ready for data
    s.send(PONG)

    logging.info("Receiving for cluster specification")

    while True:
        data = s.recv(CHUNK_SIZE)

        if not data:
            break

        chunks.append(data)

except:
    logging.error("Exception")
    raise

finally:
    logging.info('Closing {a}'.format(a=address))
    s.shutdown(socket.SHUT_RDWR)
    s.close()

try:
    # Join chunks into string, parse json
    # and extract cluster specification, job and task
    data = json.loads(''.join(chunks))
    cluster, job, task = (data[v] for v in ('cluster', 'job', 'task'))
except KeyError as e:
    logging.error("Key '{k}' not in dictionary".format(k=e.message))
    raise

if args.start is True:
    import tensorflow as tf
    import montblanc
    from montblanc.impl.rime.tensorflow.RimeSolver import RimeSolver

    server = tf.train.Server(cluster, job_name=job, task_index=task)
    logging.info("Server Target is '{st}'".format(st=server.target))
    logging.info("Server Job Name is '{j}'".format(j=server.server_def.job_name))
    logging.info("Server Task Index is '{t}'".format(t=server.server_def.task_index))

    import time

    slvr_cfg = {
        'tf_server_target' : server.target,
        'tf_job_name' : server.server_def.job_name,
        'tf_task_index' : server.server_def.task_index,
    }

    slvr = RimeSolver(slvr_cfg)

    logging.info("Created tensorflow solver")

    slvr.solve()

    time.sleep(1)
