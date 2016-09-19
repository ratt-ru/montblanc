import tensorflow as tf
from tensorflow.python.client import timeline

with tf.device('/cpu:0'):
    a = tf.random_normal(shape=[128*1024*1024])
    b = tf.random_normal(shape=[128*1024*1024])

with tf.device('/gpu:0'):

    c = a + b
    c = a + c
    c = b * c
    c = (a + b)/c
    c = a - c
    c = b + c
    c = c + c
    c = a / c

    # i = tf.constant(0)
    # cond = lambda i: tf.less(i, 10)
    # body = lambda i: tf.add(i, 1)

    # r = tf.while_loop(cond, body, [i])

with tf.Session() as S:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    S.run(tf.initialize_all_variables())
    
    print S.run(c, options=run_options, run_metadata=run_metadata)[:10]
    #print S.run(r)

    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline.json', 'w') as f:
        f.write(ctf)

