import tensorflow as tf
import numpy as np



# Test Loss Function optimization
if(0):
    BatchSize = 8
    seed = 23455
    COST = 9
    PROFIT = 2
    # COST = 2
    # PROFIT = 9

    # provide random numbers
    rng = np.random.RandomState(seed)
    # get a 32 rows 2 cols matrix (Here presents 32 groups of volumn/weight data)
    X = rng.rand(32,2)

    Y = [[x0+x1]for (x0,x1) in X] # get one row from X and set the class(Keep this structure in mind)
    print("X:\n",X)
    print("Y:\n",Y)

#-----The actual Neural Network------------------------------------------------
    # define feed forward
    x = tf.placeholder(tf.float32, shape=(None,2))
    y_ = tf.placeholder(tf.float32, shape=(None,1))
    w1 = tf.Variable(tf.random_normal([2,1],stddev=1, seed=1))

    y = tf.matmul(x, w1)

    # define back propagation/ Loss Function
    loss = tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*COST,(y_-y)*PROFIT))  # tf.greater(y,y_): y greater than y_
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    #train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss)
    #train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    # Train
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # print the initiate weight
        print("w1:\n",sess.run(w1))
        print("\n")

        STEPS = 20000
        for i in range(STEPS):
            start = (i*BatchSize)%32 # 0 to 31
            end = start+BatchSize
            sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
            if i%500 == 0:
                total_loss = sess.run(loss, feed_dict={x:X, y_:Y})
                print("After %d training steps, loss on all data is %g\n"%(i, total_loss))
                print("After %d training steps, w1 is:")
                print(sess.run(w1), "\n")
        # Output the parameters after training
        print("\n")
        print("Final w1 is:\n", sess.run(w1))



#  ------------------Moving Average--------------------------------------------------------
if(1):
    w1 = tf.Variable(0, dtype=tf.float32)  # The numbers start from 0
    global_step = tf.Variable(0,trainable=False)  # Train step number in this step, it should not be trained

    # Moving Average
    MOVING_AVERAGE_DECAY = 0.99    # Moving Average Decay Rate (A hyper parameter)
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    ema_op = ema.apply(tf.trainable_variables())

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        print(sess.run([w1, ema.average(w1)])) # Use ema.average to get w1's Moving Average

        sess.run(tf.assign(w1,1))
        sess.run(ema_op)
        print(sess.run([w1, ema.average(w1)]))


        sess.run(tf.assign(global_step,100))
        sess.run(tf.assign(w1,10))
        sess.run(ema_op)
        print(sess.run([w1,ema.average(w1)]))

        sess.run(ema_op)
        print(sess.run([w1,ema.average(w1)]))

        sess.run(ema_op)
        print(sess.run([w1,ema.average(w1)]))

        sess.run(ema_op)
        print(sess.run([w1,ema.average(w1)]))




