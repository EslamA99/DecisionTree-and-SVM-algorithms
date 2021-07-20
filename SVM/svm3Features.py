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
                w += alpha * (x * y[index]) - 2 * lamb * w
    return w


def predict(w, x):
    return -1 if (numpy.dot(x, w)) < 0 else 1


def accuracy(X, y, w):
    counter = 0
    for index, x in enumerate(X):
        if predict(w, x) == y[index]:
            counter += 1
    return (counter / len(y)) * 100


def runSvm3Features():
    heart_data = read_csv("heart.csv")
    exang_thalach_ca = numpy.array(heart_data[['exang', 'thalach', 'ca', 'target']])
    thal_exang_ca = numpy.array(heart_data[['thal', 'exang', 'ca', 'target']])
    exang_oldpeak_ca = numpy.array(heart_data[['exang', 'oldpeak', 'ca', 'target']])
    cp_ca_oldpeak = numpy.array(heart_data[['cp', 'ca', 'oldpeak', 'target']])
    oldpeak_thal_cp = numpy.array(heart_data[['oldpeak', 'thal', 'cp', 'target']])
    thalach_trestbps_oldpeak = numpy.array(heart_data[['thalach', 'trestbps', 'oldpeak', 'target']])

    data = [exang_thalach_ca, thal_exang_ca, exang_oldpeak_ca, cp_ca_oldpeak, oldpeak_thal_cp, thalach_trestbps_oldpeak]
    namesOfTables = ['exang_thalach_ca', 'thal_exang_ca', 'exang_oldpeak_ca', 'cp_ca_oldpeak', 'oldpeak_thal_cp',
                     'thalach_trestbps_oldpeak']
    alphaList = [1, 0.1, 0.01, 0.00003, 0.00001, 0.000001]
    iteration = 1000
    lamb = 1 / iteration
    maxArr = []
    alphamx = []
    for i in range(len(data)):
        numpy.random.shuffle(data[i])
        x = numpy.c_[data[i][:, 0], data[i][:, 1], data[i][:, 2]]
        Y = numpy.c_[data[i][:, 3]]
        print("\n",i,"- three features = ", namesOfTables[i])
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
    print("max accuracy features= ",namesOfTables[maxArr.index(max(maxArr))])




'''def visualize_svm(X, y, w, b):
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)

    x0_1 = amin(X[:, 0])
    x0_2 = amax(X[:, 0])

    x1_1 = get_hyperplane_value(x0_1, w, b, 0)
    x1_2 = get_hyperplane_value(x0_2, w, b, 0)

    x1_1_m = get_hyperplane_value(x0_1, w, b, -1)
    x1_2_m = get_hyperplane_value(x0_2, w, b, -1)

    x1_1_p = get_hyperplane_value(x0_1, w, b, 1)
    x1_2_p = get_hyperplane_value(x0_2, w, b, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'y--')
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], 'k')
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'k')

    x1_min = amin(X[:, 1])
    x1_max = amax(X[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])

    plt.show()
'''
