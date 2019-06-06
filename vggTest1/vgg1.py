# import packages
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers
import matplotlib.pyplot as plt

## Load dataset
def getData():
    # train
    dataset = np.loadtxt("COIL20_Train.csv", delimiter=',', dtype=np.float32)
    Xt = dataset[:, 0:16384]/256  # feature
    Xtrn = np.array(Xt).reshape((360, 128, 128, 1))
    #print(Xtrn.shape)
    Yt = dataset[:, 16384:] -1
    Ytrn = np_utils.to_categorical(Yt, 20)  # class label (one-hot representation)

    # test
    dataset = np.loadtxt("COIL20_Valid.csv", delimiter=',', dtype=np.float32)
    Xt = dataset[:,0:16384]/256  # feature
    Xtst = np.array(Xt).reshape((360, 128, 128, 1))
    Yt = dataset[:,16384:] - 1
    Ytst = np_utils.to_categorical(Yt, 20)  # class label (one-hot representation)

    return Xtrn, Ytrn, Xtst, Ytst

# Define VGG using function
def getVgg():

    input_tensor = keras.Input(shape=(128, 128, 1), dtype='float32', name='input')

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu', kernel_initializer='he_normal')(x)
    x = Dense(4096, activation='relu', kernel_initializer='he_normal')(x)
    output_tensor = Dense(20, activation='softmax')(x)

    myvgg = Model(input_tensor, output_tensor)

    # checkpoint = ModelCheckpoint(filepath='My_VGG_weight.hdf5',
    #                              monitor='loss',
    #                              mode='min',
    #                              save_best_only=True)

    myvgg.compile(loss='categorical_crossentropy', optimizer=optimizers.adam(lr=2e-5), metrics=['accuracy'])

    return myvgg


def plotFit(history):
    fig, loss_ax = plt.subplots()

    acc_ax = loss_ax.twinx()

    loss_ax.plot(history['loss'], 'y', label='train loss')
    #loss_ax.plot(model.history['val_loss'], 'r', label='val loss')

    acc_ax.plot(history['acc'], 'b', label='train acc')
    #acc_ax.plot(model.history['val_acc'], 'g', label='val acc')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    plt.show()


# Train and Test    
def main():

    Xtrn, Ytrn, Xtst, Ytst = getData()
    generator = ImageDataGenerator(rotation_range=10, width_shift_range=0.5, height_shift_range=0.5)
    data_flow = generator.flow(Xtrn, Ytrn, batch_size=16)

    model = getVgg()
    vgg_checkpoint = ModelCheckpoint(filepath='./vgg_check.h5', monitor='val_loss', verbose=1, save_best_only=True, period=10)
    #vgg_early_stopping = EarlyStopping(monitor='val_loss', patience=1000)

    #model.fit(Xtrn, Ytrn, epochs=1000, batch_size=100)
    history = model.fit_generator(data_flow, epochs=1000, steps_per_epoch=10)#, callbacks=[vgg_checkpoint])

    print("get score")
    scores = model.evaluate(Xtst, Ytst)
    print("Test: Accuracy {:.4}".format(scores[1]))

    print(model.summary())
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    model.save("vgg_epoch1000.h5")
    plotFit(history)

# Run code
if __name__=='__main__':
    main()