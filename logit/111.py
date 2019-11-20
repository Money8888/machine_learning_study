import logistic1
import numpy as np
frTrain=open('horseColicTraining.txt')
trainingSet=[];trainingLabels=[]
for line in frTrain.readlines():
    currLine = line.strip().split('\t')
    lineArr = []
    for i in range(len(currLine) - 1):
        lineArr.append(float(currLine[i]))
    trainingSet.append(lineArr)
    trainingLabels.append(float(currLine[-1]))
trainWeights = logistic1.stocGradAscent(np.array(trainingSet), trainingLabels, 500)
'''def classifyVector(inX,weights):
    prob=logistic1.sigmoid(sum(inX*weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0

if __name__=='__main__':
    classifyVector(np.array(lineArr),trainWeights)'''
print(trainWeights);print(np.array(lineArr))