import numpy as np
import matplotlib.pyplot as plt

def loadSimpData():                          #创建单层决策树的数据集
    dataMat=np.matrix([[1,2.1],
                       [1.5,1.6],
                       [1.3,1],
                       [1,1],
                       [2,1]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#单层决策树分类函数，dimen表示特征所在列数，threshVal表示阈值，threshIneq表示标志
    retArray=np.ones((np.shape(dataMatrix)[0],1))         #分类结果列表
    if threshIneq=='lt':
        retArray[dataMatrix[:,dimen]<=threshVal]=-1.0
    else:
        retArray[dataMatrix[:,dimen]>threshVal]=-1.0
    return retArray

def buildStump(dataArr,classLabels,D):                    #寻找数据集的最佳单层决策树，D为样本权重
    dataMatrix=np.mat(dataArr);labelMat=np.mat(classLabels).T
    m,n=np.shape(dataMatrix)
    numSteps=10.0
    bestStump={}                                          #bestStump为最佳单层决策树信息
    bestClasEst=np.mat(np.zeros((m,1)))                   #bestClasEst为最佳的分类结果
    minError=float('inf')
    for i in range(n):
        rangeMin=dataMatrix[:,i].min();rangeMax=dataMatrix[:,i].max()
        stepSize=(rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:                  #lt表示小于的情况，gt表示大于的情况
                threshVal=(rangeMin+float(j)*stepSize)   #计算阈值
                predictedVals=stumpClassify(dataMatrix,i,threshVal,inequal) #计算分类结果
                errArr=np.mat(np.ones((m,1)))            #初始化误差矩阵
                errArr[predictedVals==labelMat]=0        #分类正确的赋值为0
                weightedError=D.T*errArr
                print("从第%d的特征划分，阈值为%.2f，与阈值的大小情况为%s，权重误差为%.3f"%(i,threshVal,inequal,weightedError))
                if weightedError<minError:
                    minError=weightedError
                    bestClasEst=predictedVals.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequal
    return bestStump,minError,bestClasEst

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr=[]
    m=np.shape(dataArr)[0]
    D=np.mat(np.ones((m,1))/m)                           #初始化权重
    aggClassEst=np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst=buildStump(dataArr,classLabels,D) #构建单层决策树
        print("权重为：",D.T)
        alpha=float(0.5*np.log((1.0-error)/max(error,1e-16))) #计算弱学习算法权重alpha
        bestStump['alpha']=alpha
        weakClassArr.append(bestStump)
        print("分类结果：",classEst.T)
        expon=np.multiply(-1*alpha*np.mat(classLabels).T,classEst)
        D=np.multiply(D,np.exp(expon))
        D=D/D.sum()
        aggClassEst+=alpha*classEst
        print("aggClassEst:",aggClassEst.T)
        aggErrors=np.multiply(np.sign(aggClassEst)!=np.mat(classLabels).T,np.ones((m,1)))
        errorRate=aggErrors.sum()/m
        print("错误率为：",errorRate)
        if errorRate==0.0:
            break
        return weakClassArr,aggClassEst

def adaClassify(datToClass,classifierArr):#AdaBoost分类函数,datToClass待分类样例，classifierArr训练好的分类器
    dataMatrix=np.mat(datToClass)
    m=np.shape(dataMatrix)[0]
    aggClassEst=np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):#遍历所有分类器进行分类
        classEst=stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst+=classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return np.sign(aggClassEst)



def showDataSet(dataMat,labelMat):
    data_plus=[]                              #正样本
    data_minus=[]                             #负样本
    for i in range(len(dataMat)):
        if labelMat[i]>0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np=np.array(data_plus);data_minus_np=np.array(data_minus)
    plt.scatter(np.transpose(data_plus_np)[0],np.transpose(data_plus_np)[1])
    plt.scatter(np.transpose(data_minus_np)[0],np.transpose(data_minus_np)[1])
    plt.show()

if __name__=='__main__':
    dataArr,classLabels=loadSimpData()
    """D=np.mat(np.ones((5,1))/5)
    bestStump,minError,bestClasEst=buildStump(dataArr,classLabels,D)
    print("最佳单层决策树信息为：\n",bestStump)
    print("最小误差为：\n",minError)
    print("最佳分类结果为：\n",bestClasEst)"""
    #showDataSet(dataArr,classLabels)
    weakClassArr,aggClassEst=adaBoostTrainDS(dataArr,classLabels)
    #print(weakClassArr)
    #print(aggClassEst)
    print(adaClassify([[0,0],[5,5]],weakClassArr))

