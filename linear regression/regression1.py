import matplotlib.pyplot as plt
import numpy as np

def loadDataSet(filename):
    numFeat=len(open(filename).readline().split('\t'))-1
    xArr=[];
    yArr=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr,yArr

def standRegres(xArr,yArr):#计算回归系数
    xMat=np.mat(xArr);yMat=np.mat(yArr).T
    xTx=xMat.T*xMat
    if np.linalg.det(xTx)==0.0:
        print("矩阵为奇异矩阵，不可逆")
        return
    ws=xTx.I*(xMat.T*yMat)      #I表示求逆矩阵
    return ws

def plotDataSet():
    xArr,yArr=loadDataSet('ex0.txt')
    n=len(xArr)
    xcord=[];ycord=[]
    for i in range(n):
        xcord.append(xArr[i][1])
        ycord.append(yArr[i])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord,ycord,s=20,c='blue',alpha=0.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

def plotRegression():
    xArr, yArr = loadDataSet('ex0.txt')
    ws=standRegres(xArr,yArr)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xCopy=xMat.copy()
    xCopy.sort(0)
    yHat=xCopy*ws
    #print(yHat.T-yMat)
    #print(yMat)
    print(np.corrcoef(yHat.T,yMat))#计算两个矩阵的相关系数矩阵
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(xCopy[:,1],yHat,c='red')
    ax.scatter(xMat[:,1].flatten().A[0],yMat.flatten().A[0],s=20,c='blue',alpha=0.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()


if __name__=='__main__':
    #plotDataSet()
    plotRegression()