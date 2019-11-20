import numpy as np
import matplotlib.pyplot as plt
import types

def loadDataSet(filename):
    dataMat=[]
    fr=open(filename)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        fltLine=list(map(float,curLine)) # 转换为float类型
        dataMat.append(fltLine)
    return dataMat

def plotDataSet(filename):
    dataMat=loadDataSet(filename)
    n=len(dataMat)
    xcord=[];ycord=[]
    for i in range(n):
        xcord.append(dataMat[i][0])
        ycord.append(dataMat[i][1])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord,ycord,s=20,c='blue',alpha=0.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

def binSplitDataSet(dataSet,feature,value):#根据特征切分数据集合
    mat0=dataSet[np.nonzero(dataSet[:,feature]>value)[0],:]
    mat1=dataSet[np.nonzero(dataSet[:,feature]<=value)[0],:]
    return mat0,mat1

def regLeaf(dataSet):#生成叶节点
    return np.mean(dataSet[:,-1])

def regErr(dataSet):#误差估计函数
    return np.var(dataSet[:,-1])*np.shape(dataSet)[0]

def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):#找到数据的最佳二元切分方式函数,返回最佳切分特征和最佳切分特征值
    tolS=ops[0];tolN=ops[1] #tolS允许的误差下降值,tolN切分的最少样本数
    if len(set(dataSet[:,-1].T.tolist()[0]))==1:
        return None,leafType(dataSet)#Python中函数参数可以是另一个函数名
    m,n=np.shape(dataSet)
    S=errType(dataSet)
    bestS=float('inf');bestIndex=0;bestValue=0 #分别为最佳误差,最佳特征切分的索引值,最佳特征值
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]):
            mat0,mat1=binSplitDataSet(dataSet,featIndex,splitVal)
            if (np.shape(mat0)[0]<tolN) or (np.shape(mat1)[0]<tolN):
                continue
            newS=errType(mat0)+errType(mat1)
            if newS<bestS:
                bestIndex=featIndex
                bestValue=splitVal
                bestS=newS
    if (S-bestS)<tolS:
        return  None,leafType(dataSet)
    mat0,mat1=binSplitDataSet(dataSet,bestIndex,bestValue)
    if (np.shape(mat0)[0]<tolN) or (np.shape(mat1)[0]<tolN):
        return None,leafType(dataSet)
    return bestIndex,bestValue

def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):#构建树
    feat,val=chooseBestSplit(dataSet,leafType,errType,ops)
    if feat==None:
        return val
    retTree={}
    retTree['spInd']=feat
    retTree['spVal']=val
    lSet,rSet=binSplitDataSet(dataSet,feat,val)#创建左子树和右子树
    retTree['left']=createTree(lSet,leafType,errType,ops)
    retTree['right']=createTree(rSet,leafType,errType,ops)
    return retTree

def isTree(obj):#判断测试对象是否为一棵树，obj为对象
    return (type(obj).__name__=='dict')#树为字典类型

def getMean(tree):#对树进行塌陷处理（递归返回树的平均值）
    if isTree(tree['right']):
        tree['right']=getMean(tree['right'])
    if isTree(tree['left']):
        tree['left']=getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

def prune(tree,testData):#后剪枝
    if np.shape(testData)[0]==0:
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal']) #如果有左子树或者右子树,则切分数据集
    if isTree(tree['left']):
        tree['left']=prune(tree['left'],lSet)
    if isTree(tree['right']):
        tree['right']=prune(tree['right'],rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNoMerge=np.sum(np.power(lSet[:,-1]-tree['left'],2))+np.sum(np.power(rSet[:,-1]-tree['right'],2))#power(A[i],B[i])表示A[i]^B[i],第一个数的第二个数次方
        treeMean=(tree['left']+tree['right'])/2.0
        errorMerge=np.sum(np.power(testData[:,-1]-treeMean,2))
        if errorMerge<errorNoMerge:
            return treeMean
        else:
            return tree
    else:return tree

if __name__=='__main__':
    '''testMat=np.mat(np.eye(4))
    mat0,mat1=binSplitDataSet(testMat,1,0.5)
    print('原始集合:\n',testMat)
    print('mat0:\n',mat0)
    print('mat1:\n',mat1)'''
    '''filename='ex00.txt'
    plotDataSet(filename)'''
    #myDat=loadDataSet('ex00.txt')
    #myMat=np.mat(myDat)
    '''feat,val=chooseBestSplit(myMat,regLeaf,regErr,(1,4))
    print(feat)
    print(val)'''
    #print(createTree(myMat,regLeaf,regErr,(1,4)))
    train_filename='ex2.txt'
    train_Data=loadDataSet(train_filename)
    train_Mat=np.mat(train_Data)
    tree=createTree(train_Mat)
    tree_1=createTree(train_Mat,ops=(10000,4))#预剪枝
    print(tree)
    print(tree_1)
    test_filename='ex2test.txt'
    test_Data=loadDataSet(test_filename)
    test_Mat=np.mat(test_Data)
    print(prune(tree,test_Mat))#后剪枝
