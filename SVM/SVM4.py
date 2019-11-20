import SVM1
import SVM2
import SVM3
import numpy as np
from os import listdir

def img2vector(filename):                       #将32*32的二进制图像转化为1*1024向量
    returnVect=np.zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect

def loadImages(dirName):                        #加载图片，dirName文件夹名字
    hwLabels=[]                                  #数据标签
    trainingFileList=listdir(dirName)
    m=len(trainingFileList)
    trainingMat=np.zeros((m,1024))               #数据矩阵
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        if classNumStr==9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i,:]=img2vector('%s/%s'%(dirName,fileNameStr))
    return trainingMat,hwLabels

def testDigits(kTup=('rbf',10)):
    dataArr,labelArr=loadImages('trainingDigits')
    b,alphas=SVM3.smoP(dataArr,labelArr,200,0.0001,10,kTup)
    dataMat=np.mat(dataArr);labelMat=np.mat(labelArr).transpose()
    svInd=np.nonzero(alphas.A>0)[0]                #.A表示将数据的小数点后的数值不被编译器忽略
    sVs=dataMat[svInd]
    labelSV=labelMat[svInd]
    print("支持向量个数：%d"%np.shape(sVs)[0])
    m,n=np.shape(dataMat)
    errorCount=0
    for i in range(m):
        kernelEval=SVM3.kernelTrans(sVs,dataMat[i,:],kTup)
        predict=kernelEval.T*np.multiply(labelSV,alphas[svInd])+b
        if np.sign(predict)!=np.sign(labelArr[i]):
            errorCount+=1
    print("训练集错误率：%.2f%%"%(float(errorCount)/m))
    dataArr, labelArr = loadImages('testDigits')
    dataMat = np.mat(dataArr);
    labelMat = np.mat(labelArr).transpose()
    errorCount = 0
    m, n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = SVM3.kernelTrans(sVs, dataMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("测试集错误率：%.2f%%" % (float(errorCount) / m))

if __name__=='__main__':
    testDigits()

