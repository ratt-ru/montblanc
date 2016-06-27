import pyrap.tables as pt
import numpy as np
import tensorflow as tf
import threading

# Map MS column string types to numpy types
MS_TO_NP_TYPE_MAP = {
    'int' : np.int32,
    'float' : np.float32,
    'double' : np.float64,
    'boolean' : np.bool,
    'complex' : np.complex64,
    'dcomplex' : np.complex128
}

QUEUE_SIZE = 10

class RimeChunk(object):
    __slots__ = ['_time', '_bl', '_chan']

    def __init__(self, time, bl, chan):
        if bl == 1:
            assert time == 1

        if chan == 1:
            assert bl == 1

        self._time = time
        self._bl = bl
        self._chan = chan

    @property
    def time(self):
        return self._time
    
    @property
    def bl(self):
        return self._bl


    @property
    def chan(self):
        return self._chan

class RimeDataFeeder(object):
    def __init__(self, chunk):
        self._chunk = chunk
        self._queue = tf.FIFOQueue(QUEUE_SIZE, tuple(self._queue_dtypes()))
        self._placeholders = self._queue_placeholders()

        self._enqueue_op = self._queue.enqueue(tuple(self._placeholders))
        self._dequeue_op = self._queue.dequeue()

    @property
    def chunk(self):
        return self._chunk

    @property
    def queue(self):
        return self._queue
    
    @property
    def enqueue_op(self):
        return self._enqueue_op
    
    @property
    def dequeue_op(self):
        return self._dequeue_op
    
    @property
    def placeholders(self):
        return self._placeholders
    
    def feed(self, coordinator, session):
        raise NotImplementedError()

    def _queue_placeholders(self):
        return map(lambda (i, dt): tf.placeholder(dt,name="ph_{i}".format(i=i)),
            enumerate(self._queue_dtypes()))

    def _queue_dtypes(self):
        raise NotImplementedError()

class NumpyRimeDataFeeder(RimeDataFeeder):
    def __init__(self, chunk, arrays):
        self._arrays = arrays
        super(NumpyRimeDataFeeder, self).__init__(chunk)

    def _queue_dtypes(self):
        return [a.dtype.type for a in self._arrays.itervalues()]

MAIN_TABLE = 'MAIN'
ORDERED_MAIN_TABLE = 'ORDERED_MAIN'

# Measurement Set sub-tables
ANTENNA_TABLE = 'ANTENNA'
SPECTRAL_WINDOW_TABLE = 'SPECTRAL_WINDOW'
DATA_DESCRIPTION_TABLE = 'DATA_DESCRIPTION'
POLARIZATION_TABLE = 'POLARIZATION'

SUBTABLE_KEYS = (ANTENNA_TABLE, SPECTRAL_WINDOW_TABLE,
    DATA_DESCRIPTION_TABLE, POLARIZATION_TABLE)

REQUESTED = ['ANTENNA1', 'ANTENNA2', 'UVW', 'DATA', 'FLAG', 'WEIGHT']

def subtable_name(msname, subtable=None):
    return '::'.join((msname, subtable)) if subtable else msname

def open_table(msname, subtable=None):
    return pt.table(subtable_name(msname, subtable), ack=False)

class MSRimeDataFeeder(RimeDataFeeder):
    def __init__(self, chunk, msname):
        self._msname = msname
        self._columns = REQUESTED
        # Create dictionary of tables
        self._tables = { k: open_table(msname, k) for k in SUBTABLE_KEYS }

        # Open the main measurement set
        ms = open_table(msname)
 
        # Access individual tables
        ant, spec, ddesc, pol = (self._tables[k] for k in SUBTABLE_KEYS)

        # Sanity check the polarizations
        if pol.nrows() > 1:
            raise ValueError("Multiple polarization configurations!")

        if pol.getcol('NUM_CORR') != 4:
            raise ValueError('Expected four polarizations')

        # Number of channels per band
        chan_per_band = spec.getcol('NUM_CHAN')

        # Require the same number of channels per band
        if not all(chan_per_band[0] == cpb for cpb in chan_per_band):
            raise ValueError('Channels per band {cpb} are not equal!'
                .format(cpb=chan_per_band))

        # Number of channels equal to sum of channels per band
        self.nbands = len(chan_per_band)
        self.nchan = sum(chan_per_band)

        if ddesc.nrows() != spec.nrows():
            raise ValueError("DATA_DESCRIPTOR.nrows() "
                "!= SPECTRAL_WINDOW.nrows()")

        auto_correlations = True
        field_id = 0

        # Create a view over the MS, ordered by
        # (1) time (TIME)
        # (2) baseline (ANTENNA1, ANTENNA2)
        # (3) band (SPECTRAL_WINDOW_ID via DATA_DESC_ID)
        ordering_query = ' '.join(["SELECT FROM $ms "
            "WHERE FIELD_ID={fid} ".format(fid=field_id),
            "" if auto_correlations else "AND ANTENNA1 != ANTENNA2 ",
            "ORDERBY TIME, ANTENNA1, ANTENNA2, "
            "[SELECT SPECTRAL_WINDOW_ID FROM ::DATA_DESCRIPTION][DATA_DESC_ID]"])

        # Store the main table
        self._tables[MAIN_TABLE] = ms
        self._tables[ORDERED_MAIN_TABLE] = pt.taql(ordering_query)

        self.nrows = ms.nrows()
        self.na = ant.nrows()
        # Count distinct timesteps in the MS
        t_query = "SELECT FROM $ms ORDERBY UNIQUE TIME"
        self.ntime = pt.taql(t_query).nrows()
        # Count number of baselines in the MS
        bl_query = "SELECT FROM $ms ORDERBY UNIQUE ANTENNA1, ANTENNA2"
        self.nbl = pt.taql(bl_query).nrows()

        super(MSRimeDataFeeder, self).__init__(chunk)

    def _queue_dtypes(self):
        ms = self._tables[ORDERED_MAIN_TABLE]
        # Descriptor dtype is np.int32
        return [np.int32] + [MS_TO_NP_TYPE_MAP[ms.getcoldesc(col)['valueType']]
            for col in self._columns]

    def feed(self, coordinator, session):
        chunk = self.chunk
        ordered_ms = self._tables[ORDERED_MAIN_TABLE]

        # This query removes entries associated with different DATA_DESCRIPTORS
        # for each time and baseline. We assume that UVW coordinates
        # are the same for these entries
        uvw_ms = pt.taql('SELECT FROM $ordered_ms ORDERBY UNIQUE TIME, ANTENNA1, ANTENNA2')

        for time in xrange(0, self.ntime, chunk.time):
            time_end = min(time + chunk.time, self.ntime)

            for bl in xrange(0, self.nbl, chunk.bl):
                bl_end = min(bl + chunk.bl, self.nbl)

                startrow=time*self.nbl + bl
                nrows=chunk.time*chunk.bl

                ant1 = uvw_ms.getcol('ANTENNA1', startrow=startrow, nrow=nrows)
                ant2 = uvw_ms.getcol('ANTENNA2', startrow=startrow, nrow=nrows)
                uvw = uvw_ms.getcol('UVW', startrow=startrow, nrow=nrows)

                # Multiply up by number of bands
                startrow *= self.nbands
                nrows *= self.nbands

                data = ordered_ms.getcol('DATA', startrow=startrow, nrow=nrows)
                flag = ordered_ms.getcol('FLAG', startrow=startrow, nrow=nrows)
                weight = ordered_ms.getcol('WEIGHT', startrow=startrow, nrow=nrows)

                descriptor = np.int32([0, time, time_end, self.ntime,
                    bl, bl_end, self.nbl])

                variables = [descriptor, ant1, ant2, uvw, data, flag, weight]

                print 'Enqueueing {t} {bl}'.format(t=time, bl=bl)
                # Enqueue data from this 
                session.run(self.enqueue_op, feed_dict={ p: a
                    for p, a in zip(self.placeholders, variables) })

        # Enqueue eof
        descriptor[0] = 1
        variables = [descriptor, ant1, ant2, uvw, data, flag, weight]
        session.run(self.enqueue_op, feed_dict={ p: a
            for p, a in zip(self.placeholders, variables) })

    def close(self):
        for table in self._tables.itervalues():
            table.close()

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etraceback):
        self.close()

vis_chunk = RimeChunk(1, 91, 64)

init_op = tf.initialize_all_variables()

feeder = MSRimeDataFeeder(vis_chunk, '/home/sperkins/data/WSRT.MS')

# While loop condition
cond = lambda eof, i: tf.not_equal(eof, 1)

# While loop body
def body(eof, i):
    descriptor, ant1, ant2, uvw, data, flag, weight = feeder.queue.dequeue()
    descriptor = tf.Print(descriptor, [descriptor], message='Descriptor ', summarize=10)
    return descriptor[0], i+1

# While loop
W = tf.while_loop(cond, body, [tf.constant(0), tf.constant(0)])

config = tf.ConfigProto()
#config.operation_timeout_in_ms=10000  # for debugging queue hangs

with tf.Session(config=config) as S:
    S.run(init_op)

    C = tf.train.Coordinator()

    read_thread = threading.Thread(
        target=lambda: feeder.feed(C, S),
        name='feed-thread')
    read_thread.daemon = False
    read_thread.start()

    print 'Running while loop'
    eof, i = S.run(W)

    print eof, i

    C.join([read_thread])
    feeder.close()

feeder = NumpyRimeDataFeeder(vis_chunk, {'DATA' : np.empty(shape=(10,10),dtype=np.complex128)})