from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import regression1
import regression3

####前向逐步回归

def regularize(xMat,yMat):                  #数据标准化,标签y列的标准化直接是y减去y的平均值
    inxMat=xMat.copy()
    inyMat=yMat.copy()
    yMean=np.mean(yMat,0)
    inyMat=yMat-yMean
    inMeans=np.mean(inxMat,0)
    inVar=np.var(inxMat,0)
    inxMat=(inxMat-inMeans)/inVar
    return inxMat,inyMat


def stageWise(xArr,yArr,eps=0.01,numIt=100):#前向逐步线性回归，eps表示每次迭代需要调整的步长，numIt为迭代次数
    xMat=np.mat(xArr);yMat=np.mat(yArr).T
    xMat,yMat=regularize(xMat,yMat)
    m,n=np.shape(xMat)
    returnMat=np.zeros((numIt,n))            #numIt次迭代的回归系数矩阵
    ws=np.zeros((n,1))
    wsTest=ws.copy()
    wsMax=ws.copy()
    for i in range(numIt):
        lowestError=float('inf')            #初始化最小误差
        for j in range(n):
            for sign in [-1,1]:
                wsTest=ws.copy()
                wsTest[j]+=eps*sign
                yTest=xMat*wsTest
                rssE=regression3.rssError(yMat.A,yTest.A)
                if rssE<lowestError:
                    lowestError=rssE
                    wsMax=wsTest
        ws=wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

def plotstageWiseMat():
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    xArr,yArr = regression1.loadDataSet('abalone.txt')
    returnMat=stageWise(xArr,yArr,0.005,1000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(returnMat)
    ax_title_text = ax.set_title(u'前向逐步回归的迭代次数与回归系数的关系', FontProperties=font)
    ax_xlabel_text = ax.set_xlabel(u'迭代次数', FontProperties=font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties=font)
    plt.setp(ax_title_text, size=15, weight='bold', color='red')
    plt.setp(ax_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=10, weight='bold', color='black')
    plt.show()

if __name__=='__main__':
    plotstageWiseMat()