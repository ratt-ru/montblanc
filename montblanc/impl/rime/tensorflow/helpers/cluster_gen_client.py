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
    import numpy as np
    import time

    server = tf.train.Server(cluster, job_name=job, task_index=task)
    logging.info("Server Target is '{st}'".format(st=server.target))

    g = tf.Graph()

    with g.as_default():
        with tf.container('shared'):
            queue_in = tf.FIFOQueue(10, [tf.int32],
                name='queue_in',
                shared_name='master_queue_in')

            queue_out = tf.FIFOQueue(10, [tf.string],
                name='queue_out',
                shared_name='master_queue_out')

            tmp = tf.Variable(-1, tf.float32, name='master_tmp')

        do_deq = queue_in.dequeue()
        do_enq = queue_out.enqueue("Hello World")

    with tf.Session(server.target, graph=g) as S:
        S.run(tf.initialize_local_variables())
        print S.run([do_deq])
        print S.run([do_deq])
        print S.run([do_deq])
        print S.run([do_deq])

        print 'Value of master_tmp={mt}.'.format(mt=S.run(tmp))

        S.run(do_enq)

        time.sleep(2)


