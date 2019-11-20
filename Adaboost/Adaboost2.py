import Adaboost1
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    numFeat=len((open(fileName).readline().split('\t')))
    dataMat=[]
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

if __name__=='__main__':
    dataArr,LabelArr=loadDataSet('horseColicTraining2.txt')
    weakClassArr,aggClassEst=Adaboost1.adaBoostTrainDS(dataArr,LabelArr)
    testArr,testLabelArr=loadDataSet('horseColicTest2.txt')
    print(weakClassArr)
    predictions=Adaboost1.adaClassify(dataArr,weakClassArr)
    errArr=np.mat(np.ones((len(dataArr),1)))
    print('训练集的错误率：%.3f%%'%(float(errArr[predictions!=np.mat(LabelArr).T].sum()*100/len(dataArr))))
    predictions=Adaboost1.adaClassify(testArr,weakClassArr)
    errArr=np.mat(np.ones((len(testArr),1)))
    print('测试集的错误率：%.3f%%' % (float(errArr[predictions != np.mat(testLabelArr).T].sum() * 100 / len(testArr))))