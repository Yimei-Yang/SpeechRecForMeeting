import pandas as pd
import numpy as np
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(x_test.shape)
x_train, x_test = x_train / 255.0, x_test / 255.0
plt.imshow(x_train[8], cmap=plt.cm.binary)
