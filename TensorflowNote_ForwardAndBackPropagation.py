import tensorflow as tf
import numpy as np


# Feed Forward---------------------------------------------------------------------
# concept input and weight parameters
if(0):
    x = tf.placeholder(tf.float32,shape=(1,2)) # feed one group of data at a time, each group has two features
    w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1)) # randomly provide weights, which are 2 rows 3 cols, standard deviation=1
    w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

    # feed forward process
    a = tf.matmul(x, w1) # The first calculating process
    y = tf.matmul(a, w2) # The second
    #-------one hidden layer

    # Use Session to get the result
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print("y in this project is:\n", sess.run(y, feed_dict={x:[[0.7,0.5]]})) # feed_dict is a dict to give value to x

if(0):
    x = tf.placeholder(tf.float32, shape=(None, 2))  # Nomber of Group is not decided
    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))  # randomly provide weights, which are 2 rows 3 cols, standard deviation=1
    w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

    # feed forward process
    a = tf.matmul(x, w1)  # The first calculating process
    y = tf.matmul(a, w2)  # The second
    # -------one hidden layer

    # Use Session to get the result
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print("y in this project is:\n", sess.run(y, feed_dict={x: [[0.7, 0.5],[0.8,0.4],[0.2,0.3]]}))

# Feed Forward---------------------------------------------------------------------




# Back Propagation-----------------------------------------------------------------

if(1):
    BatchSize = 8
    seed = 23455

    # provide random numbers
    rng = np.random.RandomState(seed)
    # get a 32 rows 2 cols matrix (Here presents 32 groups of volumn/weight data)
    X = rng.rand(32,2)

    Y = [[int(x0+x1<1)]for (x0,x1) in X] # get one row from X and set the class(Keep this structure in mind)
    print("X:\n",X)
    print("Y:\n",Y)

    # define feed forward
    x = tf.placeholder(tf.float32, shape=(None,2))
    y_ = tf.placeholder(tf.float32, shape=(None,1))
    w1 = tf.Variable(tf.random_normal([2,3],stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)

    # define back propagation/ Loss Function
    loss = tf.reduce_mean(tf.square(y-y_))
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    #train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss)
    #train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    # Train
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # print the initiate weight
        print("w1:\n",sess.run(w1))
        print("w2:\n",sess.run(w2))
        print("\n")

        STEPS = 3000
        for i in range(STEPS):
            start = (i*BatchSize)%32 # 0 to 31
            end = start+BatchSize
            sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
            if i%500 == 0:
                total_loss = sess.run(loss, feed_dict={x:X, y_:Y})
                print("After %d training steps, loss on all data is %g\n"%(i, total_loss))

        # Output the parameters after training
        print("\n")
        print("w1:\n", sess.run(w1))
        print("w2:\n", sess.run(w2))
# Back Propagation-----------------------------------------------------------------