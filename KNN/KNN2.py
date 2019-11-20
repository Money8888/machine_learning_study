import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import KNN1
"""
打开并解析文件，对数据进行分类，1代表不喜欢，2代表一般，3代表极具魅力
filename-文件名
返回returnMat-特征矩阵，classLabelVector-分类Label向量
"""
def file2matrix(filename):
    fr=open(filename)                       #fr文件指针，打开文件filename
    arrayOLines=fr.readlines()              #读取文件所有内容行直到结束符EOF，返回列表
    numberOFLines=len(arrayOLines)          #得到文件行数
    returnMat=np.zeros((numberOFLines,3))   #初始化特征矩阵，3由数据集决定，表示特征数
    classLabelVector=[]                     #初始化分类向量
    index=0                                 #初始化行的索引初值为0
    for line in arrayOLines:
        line =line.strip()                      #strip()默认删除空白符(包括'\n','\r','\t',' ')
        listFromLine=line.split('\t')           #split('\t',n)根据'\t'进行分割成n+1片，没有则遇着就分割
        returnMat[index,:]=listFromLine[0:3]    #将前三列数据提取出来，存放如特征矩阵中
        if listFromLine[-1]=='didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1]=='smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1]=='largeDoses':   #将数据中的程度进行量化
            classLabelVector.append(3)
        index+=1
    return returnMat,classLabelVector

#数据归一化
def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)                     #max(0),min(0)返回矩阵中每一列的最值，max(1),min(1)返回每一行的最值
    ranges=maxVals-minVals
    normDataSet=np.zeros(np.shape(dataSet))    #shape(A)返回矩阵A的行列数，初始化标准化数据集
    m=dataSet.shape[0]
    normDataSet=dataSet-np.tile(minVals,(m,1))
    normDataSet=normDataSet/np.tile(ranges,(m,1)) #此方式比循环好，也可以用ones函数
    return normDataSet,ranges,minVals

def showdatas(datingDataMat,datingLabels):
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)#设置汉字格式,r表示不转义，即字符串中\保留
    fig,axs=plt.subplots(nrows=2,ncols=2,sharex=False,sharey=False,figsize=(13,8)) #fig画布大小为13*8，nrow和ncol表示分布在几行几列
    #fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))
    numberOFLabels=len(datingLabels)
    LabelsColors=[]
    for i in datingLabels:
        if i==1:
            LabelsColors.append('black')
        if i==2:
            LabelsColors.append('orange')
        if i==3:
            LabelsColors.append('red')           #标记各个标签的颜色
    axs[0][0].scatter(x=datingDataMat[:,0],y=datingDataMat[:,1],color=LabelsColors,s=15,alpha=0.5) #s表示散点大小为15，alpha表示透明度为0.5
    axs0_title_text=axs[0][0].set_title(u'每年获得的飞行常用里程数与玩视频游戏所消耗时间占比',FontProperties=font) #u防止中文字符串出现乱码
    axs0_xlabel_text=axs[0][0].set_xlabel(u'每年获得的飞行常用里程数',FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    plt.setp(axs0_title_text,size=9,weight='bold',color='red')
    plt.setp(axs0_xlabel_text,size=7,weight='bold',color='black')
    plt.setp(axs0_ylabel_text,size=7,weight='bold',color='black')

    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常用里程数与每周消费的冰激淋公升数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常用里程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    didntLike=mlines.Line2D([],[],color='black',marker='.',markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',markersize=6, label='largeDoses')   #设置图例

    axs[0][0].legend(handles=[didntLike, smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])   #添加图例

    plt.show()  #显示图片

def datingClassTest(filename):                                              #分类测试函数
    #filename = "datingTestSet.txt"
    datingDataMat, datingLabels = file2matrix(filename)
    hoRatio=0.1
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m=normMat.shape[0]                                               #获取标准化后的特征矩阵的行数
    numTestVecs=int(m*hoRatio)                                       #取数据集的10%作为测试集个数，int向下取整函数
    errorCount=0.0                                                   #初始化分类错误数为0

    for i in range(numTestVecs):
        classifierResult=KNN1.classfy0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],4)
        print("分类结果：%d\t真实类别：%d"%(classifierResult,datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount+=1.0
    print("错误率：%f%%"%(errorCount*100/float(numTestVecs)))         #双写百分号转义为一个百分号

def classifyPerson(inArr,filename):                                #验证一组数据的分类，这里为为他判断他对某个的喜好程度，inArr为待测向量，filename为数据集
    resultList=['讨厌','有些喜欢','非常喜欢']                       #定义结果标准向量，对应不喜欢，魅力一般，极具魅力
    datingDataMat, datingLabels = file2matrix(filename)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    norminArr=(inArr-minVals)/ranges                                #待测数据归一化
    classifierResult=KNN1.classfy0(norminArr,normMat,datingLabels,3)
    print("你可能%s这个人"%(resultList[classifierResult-1]))       #Python格式化输出格式print("%d"%(i)),



if __name__=='__main__':
    filename = "datingTestSet.txt"
    datingDataMat,datingLabels=file2matrix(filename)   #提取特征矩阵和标签向量
    normDataSet, ranges, minVals=autoNorm(datingDataMat)  #归一化特征矩阵
    #print(datingDataMat)
    #print(datingLabels)
    #print(normDataSet)
    #print(ranges)
    #print(minVals)
    showdatas(datingDataMat,datingLabels)   #显示的是特征和特征之间的关系
    datingClassTest(filename)               #测试数据
    precentTats=float(input("玩视频游戏所耗时间百分比:"))   #float转化为浮点数，a=input("abc:")运行后就是abc:+等待输入，并把输入值赋值给a
    ffMiles = float(input("每年获得的飞行常用里程数:"))     #输入的是原始数据，不用归一化
    iceCream = float(input("每周消费的冰激淋公升数:"))
    inArr=np.array([precentTats,ffMiles,iceCream])
    classifyPerson(inArr,filename)
