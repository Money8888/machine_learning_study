import regression1
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

####岭回归

def ridgeRegres(xMat,yMat,lam=0.2):#岭回归，lam为缩减系数
    xTx=xMat.T*xMat
    denom=xTx+np.eye(np.shape(xMat)[1])*lam
    if np.linalg.det(denom)==0.0:
        print("矩阵为奇异矩阵，不可逆")
        return
    ws=denom.I*(xMat.T*yMat)
    return  ws

def ridgeTest(xArr,yArr):          #岭回归测试函数
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean=np.mean(yMat,axis=0)
    yMat=yMat-yMean
    xMeans=np.mean(xMat,axis=0)
    xVar=np.var(xMat,axis=0)
    xMat=(xMat-xMeans)/xVar
    numTestPts=30                   #30个不停lam的测试
    wMat=np.zeros((numTestPts,np.shape(xMat)[1])) #初始化回归系数矩阵
    for i in range(numTestPts):
        ws=ridgeRegres(xMat,yMat,np.exp(i-10))
        wMat[i,:]=ws.T
    return wMat

def plotwMat():#绘制岭回归系数矩阵
    font=FontProperties(fname=r"c:\windows\fonts\simsun.ttc",size=14)
    abX,abY=regression1.loadDataSet('abalone.txt')
    redgeWeights=ridgeTest(abX,abY)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(redgeWeights)
    ax_title_text=ax.set_title(u'log(lambda)与回归系数的关系',FontProperties=font)
    ax_xlabel_text=ax.set_xlabel(u'log(lambda)',FontProperties=font)
    ax_ylabel_text=ax.set_ylabel(u'回归系数',FontProperties=font)
    plt.setp(ax_title_text,size=20,weight='bold',color='red')
    plt.setp(ax_xlabel_text,size=10,weight='bold',color='black')
    plt.setp(ax_ylabel_text,size=10,weight='bold',color='black')
    plt.show()

if __name__=='__main__':
    plotwMat()



