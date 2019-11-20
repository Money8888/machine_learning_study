import matplotlib.pyplot as plt
import numpy as np
import SVM1
import SVM2
import random

def showDataSet(dataMat, labelMat):
    data_plus=[]                                  #正样本
    data_minus=[]                                 #负样本
    for i in range(len(dataMat)):
        if labelMat[i]>0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np=np.array(data_plus)              #转换为numpy矩阵
    data_minus_np=np.array(data_minus)            #转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])   #正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1]) #负样本散点图
    plt.show()

def kernelTrans(X,A,kTup):                       #X为数据矩阵，A为单个数据的向量，kTup为包含核函数信息的元组，此函数将数据转换为更高维的空间
    m,n=np.shape(X)
    K=np.mat(np.zeros((m,1)))
    if kTup[0]=='lin':                            #线性核函数
        K=X*A.T
    elif kTup[0]=='rbf':                          #高斯核函数
        for j in range(m):
            deltaRow=X[j,:]-A
            K[j]=deltaRow*deltaRow.T
        K=np.exp(K/(-1*kTup[1]**2))               #计算高斯核
    return K

class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler,kTup):
        self.X=dataMatIn                                   #数据矩阵
        self.labelMat=classLabels                          #数据标签
        self.C=C                                           #松弛变量
        self.tol=toler                                     #容错率
        self.m=np.shape(dataMatIn)[0]
        self.alphas=np.mat(np.zeros((self.m,1)))           #初始化alphas
        self.b=0
        self.eCache=np.mat(np.zeros((self.m,2)))
        self.K=np.mat(np.zeros((self.m,self.m)))           #初始化核K
        for i in range(self.m):
            self.K[:,i]=kernelTrans(self.X,self.X[i,:],kTup)

def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):        #完整的SMO算法
    oS=optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler,kTup)
    iter=0
    entireSet=True
    alphaPairsChanged=0
    while (iter<maxIter) and ((alphaPairsChanged>0) or (entireSet)):
        alphaPairsChanged=0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged+=SVM2.innerL(i,oS)
                print("全样本遍历：第%d次迭代 样本：%d，alpha优化次数：%d"%(iter,i,alphaPairsChanged))
            iter+=1
        else:
            nonBoundIs=np.nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]     #遍历不在边界0和C的alpha
            for i in nonBoundIs:
                alphaPairsChanged+=SVM2.innerL(i,oS)
                print("非边界遍历：第%d次迭代 样本：%d，alpha优化次数：%d"%(iter,i,alphaPairsChanged))
            iter+=1
        if entireSet:
            entireSet=False
        elif alphaPairsChanged==0:
            entireSet=True
        print("迭代次数：%d"%iter)
    return oS.b,oS.alphas

def testRbf(k1=1.3):                            #k1使用高斯核函数时表示到达率
    dataArr,labelArr=SVM1.loadDataSet('testSetRBF.txt')          #加载训练集
    b,alphas=smoP(dataArr,labelArr,200,0.0001,100,('rbf,k1'))
    dataMat=np.mat(dataArr);labelMat=np.mat(labelArr).transpose()
    svInd=np.nonzero(alphas.A>0)[0]                              #获得支持向量
    sVs=dataMat[svInd]
    labelSV=labelMat[svInd]
    print("支持向量个数：%d"%np.shape(sVs)[0])
    m,n=np.shape(dataMat)
    errorCount=0                                                 #错误数
    for i in range(m):
        kernelEval=kernelTrans(sVs,dataMat[i,:],('rbf',k1))      #计算各个点的核
        predict=kernelEval.T*np.multiply(labelSV,alphas[svInd])+b #根据支持向量点，计算超平面，并返回预测结果
        if np.sign(predict)!=np.sign(labelArr[i]):               #sign函数为绝对值函数
            errorCount+=1
    print("训练集错误率：%.2f%%"%(float(errorCount)*100/m))       #输出错误率

    dataArr,labelArr=SVM1.loadDataSet('testSetRBF2.txt')         # 加载测试集
    errorCount=0
    datMat=np.mat(dataArr);labelMat=np.mat(labelArr).transpose()
    m,n=np.shape(datMat)
    for i in range(m):
        kernelEval=kernelTrans(sVs,datMat[i,:],('rbf',k1))  # 计算各个点的核
    predict=kernelEval.T*np.multiply(labelSV,alphas[svInd])+b  # 根据支持向量的点，计算超平面，返回预测结果
    if np.sign(predict) != np.sign(labelArr[i]):
        errorCount+=1                                       # 返回数组中各元素的正负符号，用1和-1表示，并统计错误个数
    print("测试集错误率: %.2f%%" %(float(errorCount)*100/m))

if __name__=='__main__':
    testRbf()
