# import packages
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras import optimizers
import matplotlib.pyplot as plt

## Load dataset
def getData():
    # train
    dataset = np.loadtxt("COIL20_Train.csv", delimiter=',', dtype=np.float32)
    Xtrn = dataset[:, 0:16384]/256  # feature values are 0~1
    Yt = dataset[:, 16384:] -1
    Ytrn = np_utils.to_categorical(Yt, 20)  # class label (one-hot representation)

    # test
    dataset = np.loadtxt("COIL20_Valid.csv", delimiter=',', dtype=np.float32)
    Xtst = dataset[:,0:16384]/256  # feature
    Yt = dataset[:,16384:] - 1
    Ytst = np_utils.to_categorical(Yt, 20)  # class label (one-hot representation)

    return Xtrn, Ytrn, Xtst, Ytst

# Define MLP(2-10-2) using function
def getDenseNet():
    model = Sequential()
    model.add(Dense(100, input_dim=16384, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    #model.add(Dense(100, activation='relu'))
    model.add(Dense(20, activation='softmax'))
    model.compile(loss='categorical_crossentropy',  optimizer=optimizers.adam(lr=10e-4), metrics=['accuracy'])

    return model

def plotFit(model):
    fig, loss_ax = plt.subplots()

    acc_ax = loss_ax.twinx()

    loss_ax.plot(model.history['loss'], 'y', label='train loss')
    #loss_ax.plot(model.history['val_loss'], 'r', label='val loss')

    acc_ax.plot(model.history['acc'], 'b', label='train acc')
    #acc_ax.plot(model.history['val_acc'], 'g', label='val acc')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    plt.show()



# Train and Test    
def main():
    model = getDenseNet()
    Xtrn, Ytrn, Xtst, Ytst = getData()

    modelFit = model.fit(Xtrn, Ytrn, epochs=300, batch_size=50)  # train

    print("get score")
    scores = model.evaluate(Xtst, Ytst)
    print("Test: Accuracy {:.4}".format(scores[1]))
    print(model.summary())
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    #model.save("dense50.h5")
    plotFit(modelFit)

# Run code
if __name__=='__main__':
    main()