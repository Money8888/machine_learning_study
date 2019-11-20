'''
def Gradient_Ascent_test():#梯度上升求最大值，梯度下降求最小值
    def f_prime(x_old):  # f(x)的导数
        return -2 * x_old + 4
    x_old = -1  # 给一个小于x_new的值
    x_new = 0  # 梯度上升算法初始值，即从(0,0)开始
    alpha = 0.01  # 步长，也就是学习速率，控制更新的幅度
    presision = 0.00000001  # 精度，也就是更新阈值
    while abs(x_new - x_old) > presision:
        x_old = x_new
        x_new = x_old + alpha * f_prime(x_old)  # 梯度上升公式
    print(x_new)  # 输出最终求解的极值近似值

if __name__ == '__main__':
    Gradient_Ascent_test()'''

import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.font_manager import FontProperties

def loadDataSet():                      #读取数据，并把数据分成特征集合和标签集合
    dataMat=[]
    labelMat=[]
    fr=open('testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()     #去回车，放入列表
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat,labelMat

def sigmoid(inX):                  #sigmoid函数
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn,classlabels):     #logistic的梯度上升算法
    dataMatrix=np.mat(dataMatIn)    #转换为numpy的mat格式
    labelMat=np.mat(classlabels).transpose() #转化为numpy的mat格式并进行转置
    m,n=np.shape(dataMatrix)                #返回矩阵的行列数，m行数，n列数
    alpha=0.001                             #学习效率
    maxCycles=500                           #最大迭代次数
    weights=np.ones((n,1))                  #权值向量初始化为1
    weights_array=np.array([])
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        error=labelMat-h
        weights=weights+alpha*dataMatrix.transpose()*error
        weights_array=np.append(weights_array,weights)   #将每次迭代后的权值记录在weights_array中
    weights_array=weights_array.reshape(maxCycles,n)
    return weights.getA(),weights_array                  #将矩阵转化为数组，返回权值数组
    #return weights

def stocGradAscent(dataMatrix,classlabels,numIter):  #改进的梯度上升算法，针对海量数据
    m,n=np.shape(dataMatrix)
    weights=np.ones(n)
    weights_array=np.array([])
    for j in range(numIter):
        dataIndex=list(range(m))           #样本下标
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01         #改进后的学习效率，随着迭代次数的增加而变小，但始终不等于0
            randIndex=int(random.uniform(0,len(dataIndex))) #
            h=sigmoid(sum(dataMatrix[randIndex]*weights))  #随机选取样本进行计算sigmoid函数
            error=classlabels[randIndex]-h
            weights=weights+alpha*error*dataMatrix[randIndex]
            weights_array=np.append(weights_array,weights,axis=0)  #axis=0时是加在下面，此时列数必须一致，axis=1时是加到右边，此时行数必须一致
            del(dataIndex[randIndex])      #删除已经使用的样本避免重复
    weights_array=weights_array.reshape(numIter*m,n)
    return weights,weights_array


def plotDataSet():                  #数据可视化
    dataMat,labelMat=loadDataSet()
    dataArr=np.array(dataMat)        #转变成numpy的array数组
    n=np.shape(dataMat)[0]           #数据个数
    xcord1 = [];ycord1 = []         #正样本
    xcord2 = [];ycord2 = []         #负样本
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i, 2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=20,c='red',marker='s',alpha=0.5) #正样本点样式
    ax.scatter(xcord2,ycord2,s=20,c='green',alpha=0.5)
    plt.title('DataSet')
    plt.xlabel('x');plt.ylabel('y')
    plt.show()

def plotBestFit(weights):                  #绘制分类直线即决策边界
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)  # 转变成numpy的array数组
    n = np.shape(dataMat)[0]  # 数据个数
    xcord1 = [];ycord1 = [] # 正样本
    xcord2 = [];ycord2 = []  # 负样本
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]);ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]);ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=0.5)  # 正样本点样式
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=0.5)
    x=np.arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]   #w0+w1*x+w2*y=0推出y=(-w0-w1*x)/w2
    ax.plot(x,y)
    plt.title('BestFit')
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()

def plotWeights(weights_array1,weight_array2):   #绘制两种梯度上升回归系数与迭代次数的图像
    font=FontProperties(fname=r"c:\windows\fonts\simsun.ttc",size=14)
    fig,axs=plt.subplots(nrows=3,ncols=2,sharex=False,sharey=False,figsize=(20,10))
    x1=np.arange(0,len(weights_array1),1)     #类似于matlab的linspace，1为步长
    axs[0][0].plot(x1,weights_array1[:,0])
    axs0_title_text = axs[0][0].set_title(u'改进梯度上升算法：回归系数与迭代次数关系', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0', FontProperties=font)
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')

    axs[1][0].plot(x1, weights_array1[:, 1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')

    axs[2][0].plot(x1, weights_array1[:, 2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W2', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    x2 = np.arange(0, len(weights_array2), 1)  # 类似于matlab的linspace，1为步长
    axs[0][1].plot(x2, weights_array2[:, 0])
    axs0_title_text = axs[0][1].set_title(u'梯度上升算法：回归系数与迭代次数关系', FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0', FontProperties=font)
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')

    axs[1][1].plot(x2, weights_array2[:, 1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')

    axs[2][1].plot(x2, weights_array2[:, 2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W2', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    plt.show()


if __name__=='__main__':
    #plotDataSet()
    dataMat,labelMat=loadDataSet()
    weights2,weights_array2=gradAscent(dataMat,labelMat)
    weights1,weights_array1=stocGradAscent(np.array(dataMat),labelMat,150)
    #plotBestFit(weights_array1)
    #plotWeights(weights_array1,weights_array2)
    print(type(weights2))
    #print(type(weights1))

