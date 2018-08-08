import tensorflow as tf

WIDTH = 240
HEIGHT = 120
N_CLASSES = 80

def model(features, labels, mode):
    
    x = tf.reshape(features['x'], [-1, WIDTH, HEIGHT, 1])
    
    net = tf.layers.conv2d(
        inputs = x,
        filters = 16,
        kernel_size = [2,2],
        padding = "same",
        activation = tf.nn.relu
    )

    net = tf.layers.average_pooling2d(
        inputs = net,
        pool_size = [2,2],
        strides = 2
    )

    net = tf.layers.conv2d(
        inputs = net,
        filters = 8,
        kernel_size = [2,2],
        padding = "same",
        activation = tf.nn.relu
    )

    net = tf.layers.average_pooling2d(
        inputs = net,
        pool_size = [2,2],
        strides = 2
    )

    net = tf.reshape(net, [-1, 60*30*8])
    
    dense = tf.layers.dense(
        inputs = net,
        units = 1024,
        activation = tf.nn.relu
    )

    dense = tf.layers.dropout(
        inputs = dense,
        rate = 0.4,
        training = mode == tf.estimator.ModeKeys.TRAIN
    )

    logits = tf.layers.dense(inputs = dense, units = N_CLASSES)
    
    predictions = {
        "classes": tf.argmax(input = logits, axis = 1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = model, predictions = predictions)

    onehot_labels = tf.one_hot(indices = tf.cast(labels, tf.int32), depth = N_CLASSES)
    loss = tf.losses.softmax_cross_entropy(onehot_labels = onehot_labels, logits = logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
        train = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train)

    evaluations = {
        "accuracy": tf.metrics.accuracy(labels = labels, predictions = predictions['classes'])
    }

    return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = evaluations)
