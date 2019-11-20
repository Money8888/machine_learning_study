import numpy as np
from functools import reduce
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],                #切分的词条
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]      #类别标签向量，1表示侮辱性，0表示正常
    return postingList,classVec

def setOfWords2Vec(vocablist,inputSet):  #将inputSet向量化使向量每个元素为1或者0
    returnVec =[0]*len(vocablist)         #创建一个元素都为0的向量，维度为vocablist的大小
    for word in inputSet:
        if word in vocablist:
            returnVec[vocablist.index(word)]=1 #如果词条存在词汇表vocablist中则置1
        else:
            print("词汇不在字典中！")
        return returnVec

def createVocablist(dataSet):                #返回不重复的词条列表（词汇表）
    vocabSet=set([])                           #创建一个空的不重复列表
    for document in dataSet:
        vocabSet=vocabSet | set(document)     #取并集
    return list(vocabSet)

def trainNB0(trainMatrix,trainCategory):    #已进行拉普拉斯平滑（加1平滑）和取对数解决下溢出处理
    numTrainDocs=len(trainMatrix)            #计算文档篇数
    numWords=len(trainMatrix[0])             #计算每篇文档词条数
    pAbuse=sum(trainCategory)/float(numTrainDocs)  #文档属于侮辱类的概率
    p0Num=np.ones(numWords);p1Num=np.ones(numWords)
    p0Denom=2.0;p1Denom=2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect=np.log(p1Num/p1Denom);p0Vect=np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbuse                   #返回返回类别为1和0条件下侮辱的条件概率和文档属于侮辱的全局概率

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1): #朴素贝叶斯分类函数
    #p1=reduce(lambda x,y:x*y,vec2Classify*p1Vec)*pClass1 #reduce使用lambda匿名函数表示所有数按照lambda中x和y的关系进行运算，此处为累乘
    #p0=reduce(lambda x,y:x*y,vec2Classify*p0Vec)*(1.0-pClass1)
    p1=sum(vec2Classify*p1Vec)+np.log(pClass1)   #对数真数相乘等于对数相加
    p0=sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)
    #print('p0:',p0)
    #print('p1:',p1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB(testEntry):
    postingList,classVec = loadDataSet()          #返回切分的词条和标签向量
    #print('postingList:\n', postingList)
    myVocablist=createVocablist(postingList)      #返回不重复的词汇表
    #print('myVocabList:\n', myVocablist)
    trainMat=[]
    for postinDoc in postingList:
        trainMat.append(setOfWords2Vec(myVocablist,postinDoc)) #量化矩阵
    #print('trainMat:\n', trainMat)
    p0V,p1V,pAb=trainNB0(trainMat,classVec)
    #print(p0V);print(p1V);print(pAb);
    #testEntry = ['love', 'my', 'dalmation']  # 测试样本，可更改
    '''for each in postingList:
        print(each)
    print(classVec)#逐行输出'''
    thisDoc=np.array(setOfWords2Vec(myVocablist,testEntry))
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print('属于侮辱类')
    else:
        print('属于不侮辱类')
'''if __name__=='__main__':
    #testEntry = ['love', 'my', 'dalmation']  # 测试样本，可更改
    testEntry=['stupid','garbage']
    testingNB(testEntry)'''