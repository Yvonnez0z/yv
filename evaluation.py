'''

@author: moyuliu

'''
import scipy.io as scio
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, losses, activations, Model, optimizers, metrics
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
best_val_mae_output = sys.float_info.max


class MyModel():
    def __init__(self, input_shape=(500, 56, 2)):
        inputs = layers.Input(shape=input_shape)

        self.cnn_layers = self.create_cnn(32, (5, 5), (1, 1), (64, 16), (32, 8)) + \
                          self.create_cnn(64, (3, 3), (1, 1), (32, 4), (16, 4)) + \
                          self.create_cnn(128, (2, 2), (1, 1), (4, 2), (8, 2))
        flatten_layer = layers.Flatten()
        self.branch_1 = [*self.cnn_layers, flatten_layer, *self.create_fc(), layers.Dense(1, name='heart')]
        self.branch_2 = [*self.cnn_layers, flatten_layer, *self.create_fc(), layers.Dense(1, name='breath')]
        outputs_1 = outputs_2 = inputs
        for layer in self.branch_1:
            outputs_1 = layer(outputs_1)
        for layer in self.branch_2:
            outputs_2 = layer(outputs_2)
        self.net = Model(inputs, [outputs_1, outputs_2])

    @staticmethod
    def create_cnn(filters=64, kernel_size=(2, 2), kernel_strides=(1, 1), pool_size=(2, 2), pool_strides=(1, 1)):
        module = [
            layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=kernel_strides, padding='same',
                          activation=activations.relu),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding='same')
        ]
        return module

    @staticmethod
    def create_fc(units=(128, 64, 32)):
        module = [layers.Dense(x, activation=activations.relu) for x in units]
        return module
def evaluation():
    z=np.load("test_data.npz")
    data = z['arr_0']
    data1 = z['arr_2']
    data2 = z['arr_1']
    
    sample = tf.data.Dataset.from_tensor_slices(data).map(
        lambda x: tf.cast((x - tf.reduce_mean(x)) / tf.math.reduce_std(x), tf.float32)).batch(4)
    model = MyModel()
    model.net.load_weights('./checkpoint1/optimal').expect_partial()
    data1_pred, data2_pred = model.net.predict(sample)

    data1_mape = np.mean(np.abs(data1_pred - data1) / data1)
    data2_mape = np.mean(np.abs(data2_pred - data2) / data2)
    print(data1_mape, data2_mape)
    
    data1_mae=np.mean(np.abs(data1_pred - data1) )
    data2_mae=np.mean(np.abs(data2_pred - data2) )
    print(data1_mae, data2_mae)
    
    plt.style.use("ggplot")    
    plt.figure()
    #plt.title('val_output_mae')
    plt.xlabel("The Number of Data #")
    plt.ylabel("Value (bpm)")
    plt.plot(data1_pred, label="Heart Predict")
    plt.plot(data1, 'g--', label="Heart True")
    plt.plot(data2_pred, label="Breath Predict")
    plt.plot(data2, 'r--', label="Breath True")
    plt.legend()
    plt.show()
    

    hr = np.mean(data1_pred)
    br = np.mean(data2_pred)
    b= np.mean(data2)
    h= np.mean(data1)
    print(hr, h,br,b)
    print(br,b)
    