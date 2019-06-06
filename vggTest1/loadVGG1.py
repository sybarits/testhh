import numpy as np
from keras.models import Model
from keras.utils import np_utils
from keras.models import load_model
from keras import optimizers



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









# Train and Test
def main():

    Xtrn, Ytrn, Xtst, Ytst = getData()

    # 2. 모델 불러오기

    model = load_model('vgg.h5')
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.adam(lr=2e-5), metrics=['accuracy'])

    print("get score")
    scores = model.evaluate(Xtst, Ytst)
    print("Test: Accuracy {:.4}".format(scores[1]))

    print(model.summary())
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    model.save("vgg.h5")
    #plotFit(history)

# Run code
if __name__=='__main__':
    main()