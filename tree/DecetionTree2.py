import pandas as pd
import numpy as np
import random

columnHeader = ['out', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15',
                'x16']
dataset = pd.read_csv('house-votes-84.data.txt', header=None)

dataset.columns = columnHeader
uniquesArr = []
for head in columnHeader:
    unique, _ = np.unique(dataset.get(head), return_counts=True)
    uniquesArr.append(unique)


def updateMissedData(currDataSet):
    for index, row in currDataSet.iterrows():
        tmp = np.array(row)
        yCount = 0
        nCount = 0
        tmpIndexes = []

        for i in range(len(tmp)):
            if tmp[i] == 'y':
                yCount += 1
            elif tmp[i] == 'n':
                nCount += 1
            elif tmp[i] == '?':
                tmpIndexes.append(i)

        for i in tmpIndexes:
            if yCount >= nCount:
                tmp[i] = 'y'
            else:
                tmp[i] = 'n'
        currDataSet.loc[index] = tmp
        # return currDataSet


def getTestandTrainingData(data, percent):
    trainingData = data.sample(frac=percent)
    testingData = data.drop(trainingData.index)
    return trainingData, testingData


class Node:
    def __init__(self, futureIndex, outNodeValues):

        self.futureIndex = futureIndex
        self.outNodeValues = outNodeValues
        self.branches = []
        self.childrenNodes = []
        self.entropy = 0
        self.informationGain = 0

    def calculateEntropyAndInfo(self):
        self.childrenNodes = [None] * len(self.branches)
        count = 0
        sum = 0
        for val in self.outNodeValues:
            sum = sum + val
            if val != 0:
                count = count + 1
        if count == 1:
            self.entropy = 0
        else:
            entropy = 0
            for val in self.outNodeValues:
                entropy = entropy + (-1 * (val / sum) * np.log2(val / sum))
            self.entropy = entropy
            self.informationGain = self.entropy
            for branch in self.branches:
                self.informationGain = self.informationGain - (branch.uniqueOutArrValuesSum / sum) * branch.entropy


class BranchData:  # y or no or ?
    def __init__(self, name, uniqueOutArrValues):
        self.name = name
        self.uniqueOutArrValues = uniqueOutArrValues
        self.uniqueOutArrValuesSum = 0
        self.entropy = 0

    def calculateEntroby(self):
        count = 0
        zeroCount = 0
        sum = 0
        # print(self.name, self.uniqueOutArrValues)
        for val in self.uniqueOutArrValues:
            sum = sum + val
            if val != 0:
                count = count + 1
            else:
                zeroCount = zeroCount + 1

        if count == 1 or zeroCount == len(self.uniqueOutArrValues):
            self.entropy = 0
        else:
            entropy = 0
            for val in self.uniqueOutArrValues:
                self.uniqueOutArrValuesSum = self.uniqueOutArrValuesSum + val
                entropy = entropy + (-1 * (val / sum) * np.log2(val / sum))
            self.entropy = entropy


def getMaxInfoGainNode(currDataSet):
    nodes = []
    for i in range(len(columnHeader)):
        if i == 0:
            continue
        outNodeValues = []
        for uniqueOutVal in uniquesArr[0]:
            tmpDataSet = currDataSet[currDataSet.get(columnHeader[0]) == uniqueOutVal]
            outNodeValues.append(tmpDataSet.shape[0])
        node = Node(i, outNodeValues)
        for uniqueVal in uniquesArr[i]:
            filteredDataSet = currDataSet[currDataSet.get(columnHeader[i]) == uniqueVal]
            uniqueOutArrValues = []

            for uniqueOutVal in uniquesArr[0]:
                filteredOutDataSet = filteredDataSet[filteredDataSet.get(columnHeader[0]) == uniqueOutVal]
                uniqueOutArrValues.append(filteredOutDataSet.shape[0])
            branch = BranchData(uniqueVal, uniqueOutArrValues)
            branch.calculateEntroby()
            node.branches.append(branch)
        node.calculateEntropyAndInfo()
        nodes.append(node)
    maxInfoIndex = 0
    maxInfoValue = -0.0
    for i in range(len(nodes)):
        # print(nodes[i].futureIndex, nodes[i].informationGain)
        if nodes[i].informationGain > maxInfoValue:
            maxInfoIndex = i
            maxInfoValue = nodes[i].informationGain
    maxNode = nodes[maxInfoIndex]
    return maxNode


def getRoot(trainDataSet):
    root = getMaxInfoGainNode(trainDataSet)
    # print(len(root.branches))
    root = buildTree(root, trainDataSet)

    return root


def buildTree(root, currDataSet):
    if not root:
        return root
    if currDataSet.shape[0] == 0:
        return root
    for i in range(len(root.branches)):
        if root.branches[i].entropy != 0:
            tmpDataSet = currDataSet[currDataSet.get(columnHeader[root.futureIndex]) == root.branches[i].name]
            if root.branches[i].entropy != 1 and tmpDataSet.shape[0] != len(unique[0]):
                root.childrenNodes[i] = getMaxInfoGainNode(tmpDataSet)
                root.childrenNodes[i] = buildTree(root.childrenNodes[i], tmpDataSet)
    return root


def getAccurecy(root, testData):
    match = 0
    for index, row in testData.iterrows():

        predictedValue = predict(root, row)
        # print(predictedValue)
        if predictedValue == row[0]:
            match += 1
    return (match / testData.shape[0]) * 100


def getResult(brachData):
    maxIndex = 0
    maxValue = -999
    for i in range(len(brachData.uniqueOutArrValues)):
        if brachData.uniqueOutArrValues[i] >= maxValue:
            maxValue = brachData.uniqueOutArrValues[i]
            maxIndex = i

    return uniquesArr[0][maxIndex]


def predict(root, row):
    # self.branches = []
    # self.childrenNodes = []
    if root:
        for i in range(len(root.branches)):
            if row[root.futureIndex] == root.branches[i].name:
                if root.branches[i].entropy == 0:
                    return getResult(root.branches[i])
                return predict(root.childrenNodes[i], row)
    else:
        return uniquesArr[0][0]


def treeSize(node):
    count = 1
    for i in range(len(node.branches)):
        if node.branches[i]:
            if node.childrenNodes[i]:
                count = count + treeSize(node.childrenNodes[i])
            else:
                count = count + 1

    return count


for i in range(5):
    trainingData, testingData = getTestandTrainingData(dataset, 25 / 100)
    root = getRoot(trainingData)
    print("Accuracy\n", getAccurecy(root, testingData))
    print("Tree Size\n", treeSize(root))

updateMissedData(dataset)
uniquesArr = []
for head in columnHeader:
    unique, _ = np.unique(dataset.get(head), return_counts=True)
    uniquesArr.append(unique)
percentage = 30
while percentage <= 70:
    print('******************************************************')
    accurecyArr = []
    treeHeightArr = []
    print('Percentage\n', percentage)
    for i in range(5):
        trainingData, testingData = getTestandTrainingData(dataset, percentage / 100)
        root = getRoot(trainingData)
        accurecy = getAccurecy(root, testingData)
        print("Accuracy\n", accurecy)
        heightValue = treeSize(root)
        print("Tree Size\n", heightValue)
        accurecyArr.append(accurecy)
        treeHeightArr.append(heightValue)
    percentage += 10
    print('---------------------------------------')
    minAccurecy = 999
    maxAcurrecy = -999
    meanAccuecy = 0
    for i in accurecyArr:
        meanAccuecy += i
        if i > maxAcurrecy:
            maxAcurrecy = i
        if i < minAccurecy:
            minAccurecy = i
    meanAccuecy = meanAccuecy / len(accurecyArr)
    print('min accurecy\n', minAccurecy)
    print('max accurecy\n', maxAcurrecy)
    print('mean accurecy\n', meanAccuecy)

    print('---------------------------------------')
    minHeight = 999
    maxHeight = -999
    meanHeight = 0
    for i in treeHeightArr:
        meanHeight += i
        if i > maxHeight:
            maxHeight = i
        if i < minHeight:
            minHeight = i
    meanHeight = meanHeight / len(treeHeightArr)
    print('min Size\n', minHeight)
    print('max Size\n', maxHeight)
    print('mean Size\n', meanHeight)
