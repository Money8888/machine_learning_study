import decisiontree
import pickle

##递归分类算法，直到遍历到叶子节点
def classify(inputTree,featLabels,testVec):    #inputTree表示已生成的决策树，featlabels存储选择的最优特征标签，testVec
    firstStr=next(iter(inputTree))
    #print(firstStr)
    secondDict=inputTree[firstStr]
    #print(secondDict)
    featIndex=featLabels.index(firstStr)        #index返回firstStr在featlabels列表中的序号（索引）
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel=secondDict[key]
    return classLabel

def storeTree(inputTree,filename):            #决策树的存储，存储在磁盘上
    with open(filename,'wb') as fw:
        pickle.dump(inputTree,fw)
    '''fw=open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()'''

def grabTree(filename):                       #读取决策树
    fr=open(filename,'rb')
    return pickle.load(fr)

if __name__=='__main__':
    dataSet,labels=decisiontree.createDataSet()
    featLabels=[]
    myTree=decisiontree.createTree(dataSet,labels,featLabels)
    '''testVec=[0,0]
    result=classify(myTree,featLabels,testVec)
    if result == 'yes':
        print('放贷')
    if result == 'no':
        print('不放贷')'''#分类测试
    '''storeTree(myTree,'classifierStorage.txt')
    outputTree=grabTree('classifierStorage.txt')
    print(outputTree)'''#保存和读取决策树
