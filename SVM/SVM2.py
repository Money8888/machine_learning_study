import matplotlib.pyplot as plt
import numpy as np
import random
import SVM1
import struct

class optStruct:                                        #维护所有需要操作的值的数据结构
    def __init__(self,dataMatIn,classLabels,C,toler):   #self表示类实例本身，不需要传参
        self.X=dataMatIn                                #数据矩阵
        self.labelMat=classLabels                       #数据标签
        self.C=C                                        #松弛变量
        self.tol=toler                                  #容错率
        self.m=np.shape(dataMatIn)[0]                   #数据矩阵行数
        self.alphas=np.mat(np.zeros((self.m,1)))        #初始化alphas
        self.b=0                                        #初始化偏置b为0
        self.eCache=np.mat(np.zeros((self.m,2)))        #根据矩阵行数初始化虎误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值

def calcEk(oS,k):                                      #oS为数据结构，k为标号为k的数据，函数为计算误差
    fXk=float(np.multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)+oS.b)
    Ek=fXk-float(oS.labelMat[k])
    return Ek

def selectJ(i,oS,Ei):                                  #i为第i个数据，Ei为该数据的误差,选择具有最大步长的j，返回索引和误差
    maxK=-1;maxDeltaE=0;Ej=0
    oS.eCache[i]=[1,Ei]
    validEcacheList=np.nonzero(oS.eCache[:,0].A)[0]     #返回误差不为0的索引值
    if len(validEcacheList)>1:
        for k in validEcacheList:
            if k==i:
                continue
            Ek=calcEk(oS,k)
            deltaE=abs(Ei-Ek)
            if deltaE>maxDeltaE:
                maxK=k
                maxDeltaE=deltaE
                Ej=Ek
        return maxK,Ej
    else:
        j=SVM1.selectJrand(i,oS.m)
        Ej=calcEk(oS,j)
    return j,Ej

def updateEk(oS,k):                                   #更新误差缓存
    Ek=calcEk(oS,k)
    oS.eCache[k]=[1,Ek]


def innerL(i,oS):                                     #优化的SMO算法,1表示有任意一对alpha值发生变化 0表示没有任意一对alpha值发生变化或变化太小
    Ei=calcEk(oS,i)
    if ((oS.labelMat[i]*Ei<-oS.tol) and (oS.alphas[i]<oS.C)) or ((oS.labelMat[i]*Ei>oS.tol) and (oS.alphas[i]>0)):
       j,Ej=selectJ(i,oS,Ei)
       alphaIold=oS.alphas[i].copy();alphaJold=oS.alphas[j].copy()
       if oS.labelMat[i]!=oS.labelMat[j]:
           L=max(0,oS.alphas[j]-oS.alphas[i])
           H=min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
       else:
           L=max(0,oS.alphas[j]+oS.alphas[i]-oS.C)
           H=min(oS.C,oS.alphas[j]+oS.alphas[i])
       if L==H:
           print("L==H")
           return 0
       eta=2.0 * oS.X[i,:]*oS.X[j,:].T-oS.X[i,:]*oS.X[i,:].T-oS.X[j,:]*oS.X[j,:].T
       if eta >= 0:
           print("eta>=0")
           return 0
       oS.alphas[j]-=oS.labelMat[j]*(Ei-Ej)/eta
       oS.alphas[j]=SVM1.clipAlpha(oS.alphas[j],H,L)
       updateEk(oS,j)
       if abs(oS.alphas[j]-alphaJold)<0.00001:
           print("alpha_j变化太小")
           return 0
       oS.alphas[i]+=oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
       updateEk(oS,i)
       b1=oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
       b2=oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
       if (0<oS.alphas[i]) and (oS.C>oS.alphas[i]):
           oS.b=b1
       elif (0<oS.alphas[j]) and (oS.C>oS.alphas[j]):
           oS.b=b2
       else:
           oS.b=(b1+b2)/2.0
       return 1
    else:
       return 0

def smoP(dataMatIn,classLabels,C,toler,maxIter):        #完整的SMO算法
    oS=optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler)
    iter=0
    entireSet=True
    alphaPairsChanged=0
    while (iter<maxIter) and ((alphaPairsChanged>0) or (entireSet)):
        alphaPairsChanged=0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged+=innerL(i,oS)
                print("全样本遍历：第%d次迭代 样本：%d，alpha优化次数：%d"%(iter,i,alphaPairsChanged))
            iter+=1
        else:
            nonBoundIs=np.nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]     #遍历不在边界0和C的alpha
            for i in nonBoundIs:
                alphaPairsChanged+=innerL(i,oS)
                print("非边界遍历：第%d次迭代 样本：%d，alpha优化次数：%d"%(iter,i,alphaPairsChanged))
            iter+=1
        if entireSet:
            entireSet=False
        elif alphaPairsChanged==0:
            entireSet=True
        print("迭代次数：%d"%iter)
    return oS.b,oS.alphas

def calcWs(alphas,dataArr,classLabels):#计算w
    X=np.mat(dataArr);labelMat=np.mat(classLabels).T####
    m,n=np.shape(X)
    w=np.zeros((n,1))
    for i in range(m):
        w+=np.multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

if __name__=='__main__':
    dataArr,classLabels=SVM1.loadDataSet('testSet.txt')
    b,alphas=smoP(dataArr,classLabels,0.6,0.001,40)
    w=calcWs(alphas,dataArr,classLabels)
    SVM1.showClassifer(dataArr,classLabels,w,b,alphas)



