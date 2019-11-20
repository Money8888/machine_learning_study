from math import log
import operator
def createDataSet():                    #创建数据集
    dataSet=[[0, 0, 0, 0, 'no'],         #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels=['年龄', '有工作', '有自己的房子', '信贷情况']
    return dataSet,labels

def calcShannonEnt(dataSet):            #计算香农熵
    numEntires=len(dataSet)              #返回数据集行数，shape[0]也行，m=dataSet.shape[0]
    labelCounts={}                       #保存每个标签出现次数的字典，标签为键
    for featVec in dataSet:
        currentLabel=featVec[-1]        #保存数据集的标签，-1表示最后一位
        if currentLabel not in labelCounts.keys():#如果标签尚未放入字典的键，将其放入
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1     #即表示此标签个数+1
    shannonEnt=0.0
    for key in labelCounts:             #循环的是键
        prob=float(labelCounts[key]/numEntires) #计算出此时key对应的标签的概率
        shannonEnt-=prob*log(prob,2)            #log(a,2)表示2为底a的对数,此时直接求和
    return shannonEnt

def splitdataSet(dataSet,axis,value):         #将数据集根据标签类别划分区域，dataSet为待划分的数据集，axis为划分数据集的特征，value为相应特征值
    retDataSet=[]                               #初始化返回的数据集列表
    for featVec in dataSet:                   #以行开始循环遍历数据集
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]     #收集axis以前的数据变成列表
            reducedFeatVec.extend(featVec[axis+1:])  #extend函数内部为收集axis以后的数据变成列表，然后extend又把他们重新按样本组合为列表，其中列表中不含axis特征的数据
            retDataSet.append(reducedFeatVec)  #将数据集按特征组好矩阵，append([1,2],[3,4])=[1,2,[3,4]],而extend为[1,2,3,4],
    return retDataSet

def chooseBestFeatureToSplit(dataSet):       #根据信息增益选择最优特征
    numFeatures=len(dataSet[0])-1              #特征个数，dataSet[0]表示第一行
    baseEntropy=calcShannonEnt(dataSet)        #计算数据集的香农熵H(D)
    bestInfoGain=0.0                           #初始化信息增益
    bestFeature=-1                             #初始化最优特征的索引值，即第几个特征
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet] #列表解析法求第i列所有数据，即特征i对应的列向量
        #featList=[dataSet[:i]]                       #dataSet[:,i]是数组array的读取方式，不是列表
        uniqueVals=set(featList)                     #寻找列中不重复的元素，并返回其所组成的向量
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitdataSet(dataSet,i,value) #将数据集以value进行划分为小数据集
            prob=len(subDataSet)/float(len(dataSet)) #被划分的区域占整个区域的概率
            newEntropy+=prob*calcShannonEnt(subDataSet) #累加计算i特征的条件熵
        infoGain=baseEntropy-newEntropy              #i特征的增益值
        #print("第%d个特征的增益为%.3f"%(i,infoGain))
        if(infoGain>bestInfoGain):                   #找出最大增益和最大增益对应的特征
            bestInfoGain=infoGain
            bestFeature = i
            #print(i)
    return bestFeature

def majorityCnt(classList):                        #统计各个标签下出现最多的元素，classList为标签列表
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)  #根据字典的值进行降序
    return sortedClassCount[0][0]
"""
决策树ID3递归算法
生成树的格式为：{'有自己的房子': {0: {'有工作': {0: 'no', 1: 'yes'}}, 1: 'yes'}}一个字典
"""

def createTree(dataSet,labels,featLabels):#dataSet为数据集，labels为分类标签，featLabels为选择的最优特征标签
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])== len(classList): #classList.count(classList[0])表示在列表中统计第一个元素的个数，若等于列表长度，则停止继续
        return classList[0]
    if len(dataSet[0])==1 or len(labels)==0: #dataSet[0]表示dataSet第一行
        return majorityCnt(classList)#遍历完所有特征后，返回出现次数最多的类标签
    bestFeat=chooseBestFeatureToSplit(dataSet)       #选择最优特征
    bestFeatLabel=labels[bestFeat]
    featLabels.append(bestFeatLabel)
    myTree={bestFeatLabel:{}}                         #根据最优特征生成树
    del(labels[bestFeat])                           #删除标签列表中已经使用的标签
    featValues=[example[bestFeat] for example in dataSet] #得到该最优特征属性的一列值
    uniqueVals=set(featValues)                        #去掉重复的数值
    for value in uniqueVals:                          #根据最优特征对应的不同属性值进行循环，每次循环体调用一次递归函数，递归和c语言类似
        myTree[bestFeatLabel][value]=createTree(splitdataSet(dataSet,bestFeat,value),labels,featLabels)
    return myTree


if __name__=='__main__':
    dataSet,labels=createDataSet()
    #print(dataSet)
    #print(calcShannonEnt(dataSet))
    #print(splitdataSet(dataSet, 0, 1))
    #print("最优特征索引值:" + str(chooseBestFeatureToSplit(dataSet)))
    featLabels=[]    #初始时，最优特征为空
    myTree=createTree(dataSet,labels,featLabels)
    print(myTree)
    print(featLabels)
