import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import random
import types

def loadDataSet(filename):  #读取文本中的数据集和标签集
    dataMat=[]
    labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):                      #i表示参数，m表示参数个数
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):                     #aj为参数值，H为参数上限，L为参数下限,用于调整大于H和小于L的alpha值
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

def smoSimple(dataMatIn,classLabels,C,toler,maxIter):  #简化版SMO算法，dataMatIn数据矩阵，classLabels数据标签，C惩罚因子，toler松弛变量，maxIter最大迭代次数
    dataMatrix=np.mat(dataMatIn);labelMat=np.mat(classLabels).transpose() #转化为numpy的mat存储
    b=0                                                 #初始化b参数
    m,n=np.shape(dataMatrix)                            #获得dataMatrix的维度
    alphas=np.mat(np.zeros((m,1)))                      #初始化alpha参数
    iter_num=0                                          #初始化迭代次数
    while(iter_num<maxIter):
        alphaPairsChanged=0
        for i in range(m):
            fXi=float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b
            Ei=fXi-float(labelMat[i])
            if ((labelMat[i]*Ei<-toler) and (alphas[i]<C)) or ((labelMat[i]*Ei>toler) and (alphas[i]>0)):
                j=selectJrand(i,m)
                fXj=float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b
                Ej=fXj-float(labelMat[j])
                alphaIold=alphas[i].copy();alphaJold=alphas[j].copy()
                if (labelMat[i]!=labelMat[j]):
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])
                if L==H:
                    print("L==H")
                    continue
                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
                if eta>0:
                    print("eta>0")
                    continue
                alphas[j]-=labelMat[j]*(Ei-Ej)/eta
                alphas[j]=clipAlpha(alphas[j],H,L)
                if (abs(alphas[j]-alphaJold)<0.00001):
                    print("alpha_j变化太小")
                    continue
                alphas[i]+=labelMat[j]*labelMat[i]*(alphaJold-alphas[j])
                b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0<alphas[i]) and (C>alphas[i]):
                    b=b1
                elif (0<alphas[j]) and (C>alphas[j]):
                    b=b2
                else:
                    b=(b1+b2)/2.0
                alphaPairsChanged+=1        #统计优化次数
                print("第%d次迭代 样本：%d，alpha优化次数：%d"%(iter_num,i,alphaPairsChanged))
        if (alphaPairsChanged==0):
            iter_num+=1
        else:
            iter_num=0
        print("迭代次数：%d"% iter_num)
    return b,alphas

def get_w(dataMat,labelMat,alphas):#计算w
    alphas=np.array(alphas);dataMat=np.array(dataMat);labelMat=np.array(labelMat)
    w=np.dot((np.tile(labelMat.reshape(1,-1).T,(1,2))*dataMat).T,alphas)
    return w.tolist()

def showClassifer(dataMat,labelMat,w,b,alphas):  #w分类超平面法向量，b分类超平面截距
    data_plus=[]             #正样本
    data_minus=[]            #负样本
    for i in range(len(dataMat)):
        if labelMat[i]>0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np=np.array(data_plus)         #列表转化为numpy矩阵
    data_minus_np=np.array(data_minus)
    plt.scatter(np.transpose(data_plus_np)[0],np.transpose(data_plus_np)[1],s=30,alpha=0.7)
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1],s=30,alpha=0.7)
    x1=max(dataMat)[0]
    x2=min(dataMat)[0]
    a1,a2=w
    b=float(b)
    a1=float(a1[0]);a2=float(a2[0])
    y1=(-b-a1*x1)/a2;y2=(-b-a1*x2)/a2
    plt.plot([x1,x2],[y1,y2])
    for i,alpha in enumerate(alphas):#画出支持向量点,enumerate返回索引和数值，常用
        if alpha>0:
            x,y=dataMat[i]
            plt.scatter([x],[y],s=150,c='none',alpha=0.7,linewidths=1.5,edgecolors='red')
    plt.show()



if __name__=='__main__':
    dataMat,labelMat=loadDataSet('testSet.txt')
    b,alphas=smoSimple(dataMat,labelMat,0.6,0.001,40)
    w=get_w(dataMat,labelMat,alphas)
    showClassifer(dataMat,labelMat,w,b,alphas)

