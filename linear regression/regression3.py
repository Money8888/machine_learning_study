import regression1
import regression2
import numpy as np

def rssError(yArr,yHatArr):#平方误差评价函数，yArr为真实数据，yHatArr为预测数据
    return ((yArr-yHatArr)**2).sum()

if __name__=='__main__':
    abX,abY=regression1.loadDataSet('abalone.txt')
    print('训练集与测试集相同:局部加权线性回归,核k的大小对预测的影响:')
    yHat01=regression2.lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
    yHat1=regression2.lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
    yHat10=regression2.lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)
    print('k=0.1时,误差大小为:', rssError(abY[0:99],yHat01.T))
    print('k=1  时,误差大小为:', rssError(abY[0:99],yHat1.T))
    print('k=10 时,误差大小为:', rssError(abY[0:99],yHat10.T))
    print(' ')
    print('训练集与测试集不同:局部加权线性回归,核k的大小是越小越好吗？更换数据集,测试结果如下:')
    yHat01 = regression2.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    yHat1 = regression2.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    yHat10 = regression2.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    print('k=0.1时,误差大小为:', rssError(abY[100:199], yHat01.T))
    print('k=1  时,误差大小为:', rssError(abY[100:199], yHat1.T))
    print('k=10 时,误差大小为:', rssError(abY[100:199], yHat10.T))
    print(' ')
    print('训练集与测试集不同:简单的线性归回与k=1时的局部加权线性回归对比:')
    print('k=1时,误差大小为:',rssError(abY[100:199],yHat1.T))
    ws=regression1.standRegres(abX[0:99],abY[0:99])
    yHat=np.mat(abX[100:199])*ws
    print('简单的线性回归误差大小:',rssError(abY[100:199],yHat.T.A))

