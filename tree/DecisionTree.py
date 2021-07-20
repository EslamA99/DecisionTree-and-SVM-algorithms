import pandas as pd
import numpy as np
import random

columnHeader = ['out', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15',
                'x16']
dataset = pd.read_csv('house-votes-84.data.txt', header=None)

dataset.columns = columnHeader

for index, row in dataset.iterrows():
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
    dataset.loc[index] = tmp


def getTestandTrainingData(data, percent):
    size = int(data.shape[0] * percent)
    startIndex = random.randrange(0, data.shape[0] - size)
    endIndex = startIndex + size

    trainingData = data[startIndex: endIndex]
    testingData = pd.concat([data[:startIndex], data[endIndex:]], axis=0)
    return trainingData, testingData


# print(dataset[dataset.get('x3') == 'y'])


class Node:
    def __init__(self, futureIndex, numOfDemocrat, numOfRepublican, leftData, rightData, leftNode, rightNode):
        self.futureIndex = futureIndex
        self.leftNode = leftNode
        self.rightNode = rightNode
        self.numOfDemocrat = numOfDemocrat
        self.numOfRepublican = numOfRepublican
        self.leftData = leftData
        self.rightData = rightData
        sum = numOfDemocrat + numOfRepublican
        if self.numOfDemocrat == 0:
            self.entropy = 0
        elif self.numOfRepublican == 0:
            self.entropy = 0
        else:
            self.entropy = -(numOfDemocrat / sum) * np.log2(numOfDemocrat / sum) - (numOfRepublican / sum) * np.log2(
                numOfRepublican / sum)
        self.informationGain = self.entropy - (
                (self.leftData.numOfDemocrat + self.leftData.numOfRepublican) / sum) * self.leftData.getEntropy() - (
                                       (
                                               self.rightData.numOfDemocrat + self.rightData.numOfRepublican) / sum) * self.rightData.getEntropy()


class BranchData:  # y or no
    def __init__(self, numOfDemocrat, numOfRepublican):
        self.numOfDemocrat = numOfDemocrat
        self.numOfRepublican = numOfRepublican

    def getEntropy(self):
        sum = self.numOfDemocrat + self.numOfRepublican
        if self.numOfDemocrat == 0:
            entropy = 0
        elif self.numOfRepublican == 0:
            entropy = 0
        else:
            entropy = -(self.numOfDemocrat / sum) * np.log2(self.numOfDemocrat / sum) - (
                    self.numOfRepublican / sum) * np.log2(
                self.numOfRepublican / sum)
        return entropy


def getMaxInfoGain(currDataSet):
    nodes = []
    for i in range(len(columnHeader)):
        if i == 0:
            continue
        yesDataSetOfHeader = currDataSet[currDataSet.get(columnHeader[i]) == 'y']
        noDataSetOfHeader = currDataSet[currDataSet.get(columnHeader[i]) == 'n']

        yesAndDemocratDataSetOfHeader = yesDataSetOfHeader[yesDataSetOfHeader.get('out') == 'democrat']
        yesAndRepublicanDemocratDataSetOfHeader = yesDataSetOfHeader[yesDataSetOfHeader.get('out') == 'republican']

        noAndDemocratDataSetOfHeader = noDataSetOfHeader[noDataSetOfHeader.get('out') == 'democrat']
        noAndRepublicanDataSetOfHeader = noDataSetOfHeader[noDataSetOfHeader.get('out') == 'republican']

        yes = BranchData(yesAndDemocratDataSetOfHeader.shape[0], yesAndRepublicanDemocratDataSetOfHeader.shape[0])
        no = BranchData(noAndDemocratDataSetOfHeader.shape[0], noAndRepublicanDataSetOfHeader.shape[0])

        node = Node(i, currDataSet[currDataSet.get('out') == 'democrat'].shape[0],
                    currDataSet[currDataSet.get('out') == 'republican'].shape[0], yes, no, None, None)
        nodes.append(node)
    maxInfoIndex = 0
    maxInfoValue = -0.0
    for i in range(len(nodes)):
        if nodes[i].informationGain > maxInfoValue:
            maxInfoIndex = i
            maxInfoValue = nodes[i].informationGain
        # print(nodes[i].name, nodes[i].informationGain)
    maxNode = nodes[maxInfoIndex]
    return maxNode


def getRoot(trainDataSet):
    root = getMaxInfoGain(trainDataSet)
    # print(root.name)
    root = buildTree(root, trainDataSet)
    return root


def buildTree(root, currDataSet):
    if root is None:
        return root
    if root.entropy == 1 and root.leftData.numOfDemocrat == 1:
        return root
    if currDataSet.shape[0] == 0:
        return root
    if root.leftData.getEntropy() != 0:
        tmpDataSet = currDataSet[currDataSet.get(columnHeader[root.futureIndex]) == 'y']
        root.leftNode = getMaxInfoGain(tmpDataSet)
        root.leftNode = buildTree(root.leftNode, tmpDataSet)

    if root.rightData.getEntropy() != 0:
        tmpDataSet2 = currDataSet[currDataSet.get(columnHeader[root.futureIndex]) == 'n']
        root.rightNode = getMaxInfoGain(tmpDataSet2)
        root.rightNode = buildTree(root.rightNode, tmpDataSet2)
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
    if brachData.numOfDemocrat >= brachData.numOfRepublican:
        return 'democrat'
    else:
        return 'republican'


def predict(root, row):
    if row[root.futureIndex] == 'y':
        if root.leftData.getEntropy() == 0 or (root.leftData.getEntropy() == 1 and root.leftData.numOfDemocrat == 1):
            return getResult(root.leftData)
        else:
            if root.leftNode:
                return getResult(root.leftData)
            else:
                return predict(root.leftNode, row)
    else:
        if root.rightData.getEntropy() == 0 or (root.rightData.getEntropy() == 1 and root.rightData.numOfDemocrat == 1):
            return getResult(root.rightData)
        else:
            if root.rightNode:
                return getResult(root.rightData)
            else:
                return predict(root.rightNode, row)

    # if root.leftNode is None and root.rightNode is None:
    #     if row[root.futureIndex] == 'y':
    #         return getResult(root.leftData)
    #     else:
    #         return getResult(root.rightData)
    # if row[root.futureIndex] == 'y':
    #     return predict(root.leftNode, row)
    # else:
    #     return predict(root.rightNode, row)


def treeSize(node):
    # if node is None:
    #     return 0

    if node.leftNode is None and node.rightNode is None:
        return 2
    # size1 = 0
    if node.leftNode:
        size1 = treeSize(node.leftNode)
    else:
        size1 = 1
    # size2 = 0
    if node.rightNode:
        size2 = treeSize(node.rightNode)
    else:
        size2 = 1
    return size1 + size2 + 1
    # return treeSize(node.leftNode) + treeSize(node.rightNode) + 1


# def height(node):
#     if node is None:
#         return 0
#     else:
#         lheight = height(node.leftNode)
#         rheight = height(node.rightNode)
#         if lheight > rheight:
#             return lheight + 1
#         else:
#             return rheight + 1


# trainedData = dataset.loc[148:255, 'out':'x16']  # 25%
trainedData = dataset.loc[:dataset.shape[0]*(25/100), 'out':'x16']  # 25%
# testingData = dataset.loc[dataset.shape[0]*(25/100):, 'out':'x16']  # 25%
root = getRoot(trainedData)
print(treeSize(root))
# print("Accuracy\n", getAccurecy(root, testingData))
# print('zz')
# row=['republican','y','y','y','y','y','y','y','y','y','y','y','y','y','y','y','y']
# print(predict(root, row))
#
for i in range(5):
    trainingData, testingData = getTestandTrainingData(dataset, 25 / 100)
    root = getRoot(trainingData)
    print("Accuracy\n", getAccurecy(root, testingData))
    print("Tree Size\n", treeSize(root))

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
