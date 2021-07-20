import numpy

from pandas import read_csv


def f(y, X, w):
    return y * (numpy.dot(X, w))


def svm(X, y, alpha, iteration, lamb):
    row, col = X.shape
    w = numpy.zeros(col)
    for i in range(iteration):
        for index, x in enumerate(X):
            if f(y[index], x, w) >= 1:
                w -= alpha * (2 * lamb * w)
            else:
                w += alpha * (numpy.multiply(y[index],x)) - 2 * lamb * w
    return w


def predict(w, x):
    return -1 if (numpy.dot(x, w)) < 0 else 1


def accuracy(X, y, w):
    counter = 0
    for index, x in enumerate(X):
        if predict(w, x) == y[index]:
            counter += 1
    return (counter / len(y)) * 100


def runSvm2Features():
    heart_data = read_csv("heart.csv")

    cp_oldpeak = numpy.array(heart_data[['cp', 'oldpeak', 'target']])
    cp_exang = numpy.array(heart_data[['cp', 'exang', 'target']])
    exang_chol = numpy.array(heart_data[['exang', 'chol', 'target']])
    exang_trestbps = numpy.array(heart_data[['exang', 'trestbps', 'target']])
    oldpeak_exang = numpy.array(heart_data[['exang', 'oldpeak', 'target']])
    exang_thalach = numpy.array(heart_data[['exang', 'thalach', 'target']])
    ca_oldpeak = numpy.array(heart_data[['ca', 'oldpeak', 'target']])
    oldpeak_thalach = numpy.array(heart_data[['oldpeak', 'thalach', 'target']])
    thal_trestbps = numpy.array(heart_data[['thal', 'trestbps', 'target']])
    oldpeak_thal = numpy.array(heart_data[['oldpeak', 'thal', 'target']])

    data = [cp_oldpeak, cp_exang, exang_chol, exang_trestbps, oldpeak_exang, exang_thalach, ca_oldpeak,
            oldpeak_thalach, thal_trestbps, oldpeak_thal]
    namesOfTables = ['cp_oldpeak', 'cp_exang', 'exang_chol', 'exang_trestbps', 'oldpeak_exang', 'exang_thalach',
                     'ca_oldpeak', 'oldpeak_thalach', 'thal_trestbps', 'oldpeak_thal']
    alphaList = [1, 0.1, 0.01, 0.00001, 0.000001]
    iteration = 1000
    lamb = 1 / iteration
    maxArr = []
    alphamx = []
    for i in range(len(data)):
        numpy.random.shuffle(data[i])
        x = numpy.c_[data[i][:, 0], data[i][:, 1]]
        Y = numpy.c_[data[i][:, 2]]
        print("\n",i,"- two features = ", namesOfTables[i])
        for j in range(len(alphaList)):
            # split dataset into training and testing data
            X_train = numpy.array(x[:250])
            Y_train = numpy.array(Y[:250])
            X_test = numpy.array(x[250:])
            Y_test = numpy.array(Y[250:])
            for i in range(X_train.shape[1]):
                X_train[:, i:i + 1] = (X_train[:, i:i + 1] - X_train[:, i:i + 1].mean()) / (
                    X_train[:, i:i + 1].std())
                X_test[:, i:i + 1] = (X_test[:, i:i + 1] - X_test[:, i:i + 1].mean()) / (
                    X_test[:, i:i + 1].std())
            Y_train = numpy.where(Y_train <= 0, -1, 1)
            Y_test = numpy.where(Y_test <= 0, -1, 1)
            X_train = numpy.c_[numpy.ones(len(X_train)), X_train]
            X_test = numpy.c_[numpy.ones(len(X_test)), X_test]
            w = svm(X_train, Y_train, alphaList[j], iteration, lamb)
            print("weight= ", w)
            acc = accuracy(X_train, Y_train, w)
            print("at alpha = ", alphaList[j], "accuracy of the training data = ", acc)
            testAcc = accuracy(X_test, Y_test, w)
            alphamx.append(testAcc)
            print("at alpha = ", alphaList[j], "accuracy of the testing data = ", testAcc)
        print("\n---------------------------------------------------------------------------------")
        maxArr.append(max(alphamx))
        alphamx = []
    print("max accuracy of all files= ", max(maxArr))
    print("max accuracy features= ", namesOfTables[maxArr.index(max(maxArr))])
