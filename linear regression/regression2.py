import regression1
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np

def lwlr(testPoint,xArr,yArr,k=1.0):#使用局部加权线性回归计算回归系数w，testPoint为测试样本点，k为高斯核参数
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m=np.shape(xMat)[0]
    weights=np.mat(np.eye((m)))
    for j in range(m):
        diffMat=testPoint-xMat[j,:]
        weights[j,j]=np.exp(diffMat*diffMat.T/(-2.0* k**2))
    xTx=xMat.T*(weights*xMat)
    if np.linalg.det(xTx)==0.0:
        print("矩阵为奇异矩阵，不可逆")
        return
    ws=xTx.I*(xMat.T*(weights*yMat))   #回归系数
    return testPoint*ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m=np.shape(testArr)[0]
    yHat=np.zeros(m)
    for i in range(m):
        yHat[i]=lwlr(testArr[i],xArr,yArr,k)
    return yHat

def plotlwlrRegression():
    font=FontProperties(fname=r"c:\windows\fonts\simsun.ttc",size=14)
    xArr,yArr=regression1.loadDataSet('ex0.txt')
    yHat_1=lwlrTest(xArr,xArr,yArr,1.0)
    yHat_2=lwlrTest(xArr,xArr,yArr,0.01)
    yHat_3=lwlrTest(xArr,xArr,yArr,0.003)
    xMat=np.mat(xArr);yMat=np.mat(yArr)
    srtInd=xMat[:,1].argsort(0)
    xSort=xMat[srtInd][:,0,:]
    fig,axs=plt.subplots(nrows=3,ncols=1,sharex=False,sharey=False,figsize=(10,8))
    axs[0].plot(xSort[:,1],yHat_1[srtInd],c='red')
    axs[1].plot(xSort[:,1],yHat_2[srtInd],c='red')
    axs[2].plot(xSort[:,1],yHat_3[srtInd],c='red')
    axs[0].scatter(xMat[:,1].flatten().A[0],yMat.flatten().A[0],s=20,c='blue',alpha=0.5)
    axs[1].scatter(xMat[:,1].flatten().A[0],yMat.flatten().A[0],s=20,c='blue',alpha=0.5)
    axs[2].scatter(xMat[:,1].flatten().A[0],yMat.flatten().A[0],s=20,c='blue',alpha=0.5)
    axs0_title_text=axs[0].set_title(u'局部加权回归曲线,k=1.0',FontProperties=font)
    axs1_title_text=axs[1].set_title(u'局部加权回归曲线,k=0.01',FontProperties=font)
    axs2_title_text=axs[2].set_title(u'局部加权回归曲线,k=0.003',FontProperties=font)
    plt.setp(axs0_title_text,size=8,weight='bold',color='red')
    plt.setp(axs1_title_text,size=8,weight='bold',color='red')
    plt.setp(axs2_title_text,size=8,weight='bold',color='red')
    plt.xlabel('X')
    plt.show()

if __name__=='__main__':
    plotlwlrRegression()