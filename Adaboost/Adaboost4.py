import  numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import Adaboost.Adaboost1
import Adaboost.Adaboost2

def plotROC(predStrengths,classLabels):              #predStrengths分类器的预测强度，classLabels为类别，绘制ROC曲线
    font=FontProperties(fname=r"c:\windows\fonts\simsun.ttc",size=14)
    cur=(1.0,1.0)                                     #绘制光标的位置
    ySum=0.0                                          #计算AUC(即分类标准，ROC曲线的面积，衡量学习器优劣的标准）
    numPosClas=np.sum(np.array(classLabels)==1.0)     #统计正类的数量
    yStep=1/float(numPosClas)                         #y轴步长
    xStep=1/float(len(classLabels)-numPosClas)        #x轴步长
    sortedIndicies=predStrengths.argsort()            #预测强度排序
    fig=plt.figure()
    fig.clf()
    ax=plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index]==1.0:
            delX=0;delY=yStep
        else:
            delX=xStep;delY=0
            ySum+=cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
        cur=(cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.title('AdaBoost马疝病检测系统的ROC曲线',FontProperties=font)
    plt.xlabel('假正率',FontProperties=font)
    plt.ylabel('真正率',FontProperties=font)
    ax.axis([0,1,0,1])
    print('AUC面积为：',ySum*xStep)
    plt.show()

if __name__=='__main__':
    dataArr,LabelArr=Adaboost.Adaboost2.loadDataSet('horseColicTraining2.txt')
    weakClassArr,aggClassEst=Adaboost.Adaboost1.adaBoostTrainDS(dataArr,LabelArr,10)
    plotROC(aggClassEst.T,LabelArr)