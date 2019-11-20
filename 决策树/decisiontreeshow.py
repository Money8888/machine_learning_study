import decisiontree
from matplotlib.font_manager import FontProperties
import  matplotlib.pyplot as plt
#import jianyan

def getNumLeafs(myTree):                     #递归获取决策树叶子节点数目
    numLeafs=0                                #初始化叶子数目
    firstStr=next(iter(myTree))               #获取决策树的根节点即最优特征对应的键
    secondDict=myTree[firstStr]               #获得除去头结点的字典，如A={'有自己的房子': {0: {'有工作': {0: 'no', 1: 'yes'}}, 1: 'yes'}}变成{0: {'有工作': {0: 'no', 1: 'yes'}}, 1: 'yes'}
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict': #验证该节点是否字典，如果不是则为叶子节点，type为读取变量的类型
            numLeafs+=getNumLeafs(secondDict[key])
        else:
            numLeafs+=1
    return numLeafs

def getTreeDepth(myTree):                    #递归获取决策树深度，和数据结构中类似
    maxDepth=0                                #初始化决策树深度
    firstStr=next(iter(myTree))
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth=1+getTreeDepth(secondDict[key])
        else:
            thisDepth=1
        if thisDepth>maxDepth:
            maxDepth=thisDepth
    return maxDepth

'''def plotNode(nodeText,centerPt,parentPt,nodeType):  #绘制节点，nodeText为节点名，centerPt为文本位置，parentPt为标注的箭头位置，nodeType为节点格式
    arrow_args=dict(arrowstyle="<-")                 #dict 创建字典，键为arrowstyle，值为'<-'
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    createPlot.ax1.annotate(nodeText,xy=parentPt,xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',va="center",ha="center",bbox=nodeType,arrowprops=arrow_args,FontProperties=font)'''  #绘制节点

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")  # 定义箭头格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 设置中文字体
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)
'''def plotMidText(cntrPt,parentPt,txtString):         #标注父子节点之间的边的权值（属性）,txtString为标注的内容
    xMid=(parentPt[0]-cntrPt[0])/2.0+cntrPt[0]
    yMid=(parentPt[1]-cntrPt[1])/2.0+cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString,va="center",ha="center",rotation=30)'''

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]  # 计算标注位置
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree,parentPt,nodeTxt):              #递归绘制决策树，parentPt 标注的内容，nodeTxt节点名
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    leafNode=dict(boxstyle="round4",fc="0.8")
    numLeafs=getNumLeafs(myTree)
    depth=getTreeDepth(myTree)
    firstStr=next(iter(myTree))
    cntrPt=(plotTree.xOff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)  #取中心位置
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict=myTree[firstStr]
    plotTree.yOff=plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

def createPlot(myTree):              #创建绘制画板
    fig=plt.figure(1,facecolor='white')
    fig.clf()                         #清空fig
    axprops=dict(xticks=[],yticks=[])
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)  #去掉x轴y轴，绘制一张图
    plotTree.totalW=float(getNumLeafs(myTree))
    plotTree.totalD=float(getTreeDepth(myTree))
    plotTree.xOff=-0.5/plotTree.totalW
    plotTree.yOff=1.0
    plotTree(myTree,(0.5,1.0),'')
    plt.show()


if __name__=='__main__':
    dataSet,labels=decisiontree.createDataSet()
    featLabels=[]
    myTree=decisiontree.createTree(dataSet,labels,featLabels)
    print(myTree)
    createPlot(myTree)