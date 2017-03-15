import numpy as np
import tensorflow as tf


def one_neuron_low_level(x_train, y_train, check_value):
    # Model parameters
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)

    # Model input and output
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    y = tf.placeholder(tf.float32)

    # loss
    loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    # training loop
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init) # reset values to wrong
    for i in range(1000):
      sess.run(train, {x: x_train, y: y_train})

    # evaluate training accuracy
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

    print("f(%s) = %s" % (check_value, sess.run(linear_model, {x: check_value})))


def one_neuron_high_level(x_train, y_train, check_value):
    feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]
    estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    check_value = np.array([check_value])
    input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_train}, y_train, batch_size=128, num_epochs=None, shuffle=True)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": check_value}, num_epochs=1, shuffle=False)

    estimator.train(input_fn=input_fn, steps=1000)

    train_metrics = estimator.evaluate(input_fn=train_input_fn)
    print("train metrics: %r" % train_metrics)

    for i, p in enumerate(estimator.predict(input_fn=predict_input_fn)):
        print("f(%s) = %s" % (check_value[0], p['predictions']))


def one_neuron():
    x = [float(it) for it in range(7)]
    y = [- it * 2. + 7 for it in x]
    check_x = 100.

    one_neuron_low_level(x, y, check_x)
    one_neuron_high_level(x, y, check_x)

one_neuron()
