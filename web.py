from abc import ABC
from evaluation import evaluation
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, losses, activations, Model, optimizers, metrics
from tensorflow.keras.callbacks import LambdaCallback
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
import sys
import os
import seaborn as sns
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
best_val_mae_output = sys.float_info.max
EPOCHS=100

class MyModel():
    def __init__(self, input_shape=(1200, 56, 2)):
        inputs = layers.Input(shape=input_shape)
        self.cnn_layers = self.create_cnn(32, (5, 5), (1, 1), (40, 16), (20, 8)) + \
                          self.create_cnn(64, (3, 3), (1, 1), (20, 4), (10, 4)) + \
                          self.create_cnn(128, (2, 2), (1, 1), (12, 2), (6, 2))
        flatten_layer = layers.Flatten()
        self.branch_1 = [*self.cnn_layers, flatten_layer, *self.create_fc(), layers.Dense(4)]
        outputs_1 = outputs_2 = inputs
        for layer in self.branch_1:
            outputs_1 = layer(outputs_1)

        self.net = Model(inputs, outputs_1)

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
    def create_fc(units=(64, 32)):
        module = [layers.Dense(x, activation=activations.relu) for x in units]
        return module


def pre_processing(x, y):
    x = tf.cast((x - tf.reduce_mean(x)) / tf.math.reduce_std(x), tf.float32)
    y = tf.one_hot(tf.squeeze(y), 4)
    return x, y


def main():
    z=np.load("dataset.npz")
    data = z['arr_0']
    data1 = z['arr_1']
    print(data1.shape)
    
    db_size = data.shape[0]
    batch_size = 64
    db = tf.data.Dataset.from_tensor_slices((data, data1)).map(pre_processing).shuffle(10000)
    db_train = db.take(int(db_size * 0.8)).batch(batch_size)
    db_val = db.skip(int(db_size * 0.8)).batch(batch_size)

    model = MyModel()
    model.net.summary()
    checkpoint_filepath = './checkpoint_lr/sleep'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_categorical_accuracy',
        mode='max',
        save_best_only=True)
    model.net.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                      loss=losses.CategoricalCrossentropy(from_logits=True),
                      metrics=metrics.CategoricalAccuracy())
    H = model.net.fit(db_train, epochs=EPOCHS, validation_data=db_val, callbacks=[model_checkpoint_callback])
    
    N=np.arange(0,EPOCHS) 
    plt.style.use("ggplot")    
    plt.figure()
    #plt.title('val_output_mae')
    plt.title('Accuracy')
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.plot(N, H.history["categorical_accuracy"], label="Train acc")   
    plt.plot(N, H.history["val_categorical_accuracy"], label="Val acc")
    plt.legend()
    plt.savefig('acc.png')  
    plt.show()
    np.save("multioutput_H_1.npy",H.history)


def prediction():
    z=np.load("data.npz")
    data = z['arr_0']
    data1 = z['arr_1']
    
    sample = tf.data.Dataset.from_tensor_slices(data).map(
        lambda x: tf.cast((x - tf.reduce_mean(x)) / tf.math.reduce_std(x), tf.float32)).batch(4)
    data1 = np.squeeze(data1)
    model = MyModel()
    model.net.load_weights('./checkpoint/sleep').expect_partial()
    result1 = model.net.predict(sample)
    result1 = np.argmax(result1, axis=1)
    print(confusion_matrix(data1, result1))
    
    np.save("y_pred.npy",result1)  
    np.save("y_true.npy",data1)
    

def plot_matrix():
    class_name=['Awake','Light','Deep','REM']
    num_local = np.array(range(len(class_name)))
    y_pred = np.load("y_pred.npy")
    y_true=np.load("y_true.npy")
    a=y_pred-y_true
    
    print(y_true.shape) #(2422,)
    
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    confusionmatrixs = confusion_matrix(y_true,y_pred)
    print(confusionmatrixs)
    confusionmatrixs = confusionmatrixs.astype(np.float)
    tota = np.sum(confusionmatrixs,axis=1)
    print(tota)
    print(confusionmatrixs[0,:])
    a1=np.around(np.divide(confusionmatrixs[0,:], tota[0])*100, decimals=1)
    a2=np.around(np.divide(confusionmatrixs[1,:], tota[1])*100, decimals=1)
    a3=np.around(np.divide(confusionmatrixs[2,:], tota[2])*100, decimals=1)
    a4=np.around(np.divide(confusionmatrixs[3,:], tota[3])*100, decimals=1)
    confusionmatrixs[0,:]=a1
    confusionmatrixs[1,:]=a2
    confusionmatrixs[2,:]=a3
    confusionmatrixs[3,:]=a4
    print(confusionmatrixs)
    
    a=sns.heatmap(confusionmatrixs,annot=True,fmt='.1f', linewidth=.5)
    ax.set_xticklabels(['Awake','Light','Deep','REM'])
    ax.set_yticklabels(['Awake','Light','Deep','REM'])
    plt.title('Confusion Matrix(%)')
    plt.ylabel('True')
    plt.xlabel('Predict')
    plt.show()



def data():
   
    data = np.load("sleep963_csi.npy")
    sleep_label = np.load("sleep963_label.npy")
    data =data[400:800,:,:,:]
    sleep_label=sleep_label[400:800]
    print(data.shape)
    np.savez("dataset.npz",data,sleep_label)
    
    z=np.load("dataset.npz")
    data = z['arr_0']
    data1 = z['arr_1']
    print(data1.shape)
    
    
    
    
if __name__ == '__main__':
    #main()
    evaluation()
    plot_matrix()