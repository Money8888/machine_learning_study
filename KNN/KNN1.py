import numpy as np
import operator
"""
创建数据集
"""
def createDataSet():
    #四个二维数组
    group=np.array([[1,101],[5,89],[108,5],[115,8]])
    #数组对应标签
    labels=['爱情片','爱情片','动作片','动作片']
    return group,labels


"""
函数说明:kNN算法,分类器

Parameters:
    inX - 用于进行分类的数据(测试集)
    dataSet - 用于进行训练的数据(训练集)
    labels - 分类标签
    k - kNN算法参数,选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果
"""
def classfy0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]                   #numpy中的shape[0]返回dataSet的行数
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet   #diffMat返回的是待检测样本与数据集中每个样本的横坐标差和纵坐标差
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)              #sum()为所有元素相加，sum(axis=1)表示按行的方向相加，sum(axis=0)表示按列的方向相加，sum([1,2,3],[4,5,6],axis=1)=[6,15],而axis=0则是[5,7,9]
    distances=sqDistances**2                       #返回待测样本与数据集的距离矩阵
    sortedDistIndices=distances.argsort()          #argsort函数返回从小到大的值的序号(索引)
    classCount={}                                  #定义一个记录类别次数的字典
    for i in range(k):                            #i不需要定义
        voteIlabel=labels[sortedDistIndices[i]]    #取出距离最近的前k个数据集的标签
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1  #字典中get函数，dict.get('A',b)表示若字典dict中出现键A，则返回字典中键A对应的值，若不存在，则返回b
        #这里用来统计voteIlabel中对应的标签出现的次数
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)  #key=operator.itemgetter(1)根据字典的值进行排序，key=operator.itemgetter(0)根据字典的键进行排序，reverse降序排序字典
    #这里用来统计哪个标签中的次数最多
    return sortedClassCount[0][0]                 #返回次数最多的类别

if __name__=='__main__':
    group,labels=createDataSet()
    test=[101,20]                                 #测试集数据
    test_class=classfy0(test,group,labels,3)      #调用分类函数
    print(test_class)




