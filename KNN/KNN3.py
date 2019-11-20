import numpy as np
import operator
from os import listdir     #文件夹
from sklearn.neighbors import KNeighborsClassifier as KNN

def img2vector(filename):                    #将32*32的二进制图像转化为1*1024向量
    returnVect=np.zeros((1,1024))             #初始化向量为零向量
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()                 #读一行数据，readlines为读所有行数据
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect

def handwritingClassTest(file1dir,file2dir): #手写数字识别函数，file1dir为存储训练集数据的文件夹,file2dir为测试集数据文件夹，文件名格式为1_12,表示第一类的第十二张图片
    hwlabels=[]                               #测试集的labels
    trainingFileList=listdir(file1dir)         #返回文件夹下的所有文件名
    m=len(trainingFileList)                   #返回文件夹下文件的数目
    trainingMat=np.zeros((m,1024))            #初始化所有训练集的1024向量构成的矩阵
    for i in range(m):
        fileNameStr=trainingFileList[i]       #获得文件的名字
        classNumber=int(fileNameStr.split('_')[0])#取文件名‘_’前面的数据的第一个
        hwlabels.append(classNumber)          #类别向量
        trainingMat[i,:]=img2vector(file1dir+'/%s'%(fileNameStr))
    neigh=KNN(n_neighbors=3,algorithm='brute')  #调用sklearn工具包的KNN分类器
    neigh.fit(trainingMat,hwlabels)            #拟合模型，trainingMat为训练矩阵，hwlabels为对应的标签
    testFileList=listdir(file2dir)
    errorCount=0.0                            #错误分类计数
    mTest=len(testFileList)                   #测试集文件个数
    for i in range(mTest):
        fileNameStr=testFileList[i]
        classNumber=int(fileNameStr.split('_')[0])
        vectorUnderTest=img2vector(file2dir+'/%s'%(fileNameStr)) #%()代替字符串中的%s
        classifierResult=neigh.predict(vectorUnderTest)   #预测测试集的标签
        print("分类返回结果为%d\t真实结果为%d"%(classifierResult,classNumber))
        if(classifierResult!=classNumber):
            errorCount+=1.0
    print("总共分类错误%d个数据\n错误率为%f%%"%(errorCount,errorCount*100/mTest))

if __name__=='__main__':
    file1dir='trainingDigits'
    file2dir='testDigits'
    handwritingClassTest(file1dir,file2dir)




