import logistic1
import numpy as np

def classifyVector(inX,weights):
    prob=1.0 / (1 + np.exp(sum(sum(inX*weights))))#logistic1.sigmoid(sum(inX*weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain=open('horseColicTraining.txt')
    frTest=open('horseColicTest.txt')
    trainingSet=[];trainingLabels=[]
    weights_array = np.array([])
    for line in frTrain.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(len(currLine)-1):#(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    trainWeights,weights_array=logistic1.gradAscent(trainingSet,trainingLabels)
    errorCount=0;numTestVec=0.0
    for line in frTest.readlines():
        numTestVec+=1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(len(currLine)-1):#(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        lineArr=np.array(lineArr)
        if int(classifyVector(lineArr,trainWeights))!=int(currLine[-1]):
            errorCount+=1
    errorRate=float(errorCount)*100/numTestVec
    print("测试集错误率为: %.2f%%" % errorRate)


if __name__=='__main__':
    colicTest()

