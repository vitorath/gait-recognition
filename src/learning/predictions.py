import sys
import numpy as np
import tensorflow as tf 
from sklearn.model_selection import train_test_split

sys.path.append('./src/models')
from convolutional import model

tf.logging.set_verbosity(tf.logging.INFO)

DATA_PATH = 'data/full/spectrum-feature.npy'
MODEL_DIR = 'src/models/model_dir'

data = np.load(DATA_PATH)

data_images = np.array([image[0] for image in data])
data_labels = np.array([label[1] for label in data])

X_train, X_test, y_train, y_test = train_test_split(data_images, data_labels, test_size = 0.2, random_state=42)

tensorsToLog = {"classes":"softmax_tensor"}
loggingHook = tf.train.LoggingTensorHook(tensors=tensorsToLog, every_n_iter=10)

classifier = tf.estimator.Estimator(model_fn=model, model_dir = MODEL_DIR)

train_input = tf.estimator.inputs.numpy_input_fn(
    x = {'x': X_train},
    y = y_train,
    batch_size = 100,
    num_epochs = None,
    shuffle = True
)

classifier.train(input_fn = train_input, steps = 500,  hooks=[loggingHook])

evaluation_input = tf.estimator.inputs.numpy_input_fn(
    x = {'x': X_test},
    y = y_test,
    num_epochs = 1,
    shuffle = False
)

evaluations = classifier.evaluate(input_fn = evaluation_input)
print(evaluations)

