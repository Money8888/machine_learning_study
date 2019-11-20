import Tregression
import matplotlib.pyplot as plt
import numpy as np

def plotDataSet(filename):
    dataMat=Tregression.loadDataSet(filename)
    n=len(dataMat)
    xcord=[];ycord=[]
    for i in range(n):
        xcord.append(dataMat[i][1])
        ycord.append(dataMat[i][2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord,ycord,s=20,c='blue',alpha=0.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

if __name__=='__main__':
    filename='ex0.txt'
    #plotDataSet(filename)
    myDat=Tregression.loadDataSet(filename)
    myMat=np.mat(myDat)
    print(Tregression.createTree(myMat))
