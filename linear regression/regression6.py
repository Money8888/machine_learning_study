from bs4 import BeautifulSoup
import numpy as np
import random
import regression1
import regression3
import regression4
import regression5

"""
inFile为html文件，yr为年份，numPce玩具部件数目，origPrc原价
从页面读取数据，生成retX，retY列表
"""

def scrapePage(retX,retY,inFile,yr,numPce,origPrc):#从Html页面读取数据，生成retX和retY列表
    with open(inFile,encoding='utf-8') as f:
        html=f.read()
    soup=BeautifulSoup(html)
    i=1
    currentRow=soup.find_all('table',r="%d"%i)
    while(len(currentRow)!=0):
        currentRow=soup.find_all('table',r="%d"%i)
        title=currentRow[0].find_all('a')[1].text
        lwrTitle=title.lower()
        if (lwrTitle.find('new')>-1) or (lwrTitle.find('nisb')>-1):
            newFlag=1.0
        else:
            newFlag=0.0
        soldUnicde=currentRow[0].find_all('td')[3].find_all('span')
        if len(soldUnicde)==0:
            print("商品 #%d 没有出售"%i)
        else:
            soldPrice=currentRow[0].find_all('td')[4]
            pricrStr=soldPrice.text
            pricrStr=pricrStr.replace('$','')
            pricrStr=pricrStr.replace(',','')
            if len(soldPrice)>1:
                pricrStr=pricrStr.replace('Free shipping','')
            sellingPrice=float(pricrStr)
            if sellingPrice>origPrc*0.5:
                print("%d\t%d\t%d\t%f\t%f"%(yr, numPce, newFlag, origPrc, sellingPrice))
                retX.append([yr,numPce,newFlag,origPrc])
                retY.append(sellingPrice)
        i+=1
        currentRow=soup.find_all('table',r="%d"%i)

def setDataCollect(retX,retY):#依次读取六中套装的数据，生成数据矩阵
    scrapePage(retX, retY, 'D:\PycharmProjects\linear regression/setHtml/lego8288.html', 2006, 800, 49.99)     # 2006年的乐高8288,部件数目800,原价49.99
    scrapePage(retX, retY, 'D:\PycharmProjects\linear regression/setHtml/lego10030.html', 2002, 3096, 269.99)  # 2002年的乐高10030,部件数目3096,原价269.99
    scrapePage(retX, retY, 'D:\PycharmProjects\linear regression/setHtml/lego10179.html', 2007, 5195, 499.99)  # 2007年的乐高10179,部件数目5195,原价499.99
    scrapePage(retX, retY, 'D:\PycharmProjects\linear regression/setHtml/lego10181.html', 2007, 3428, 199.99)  # 2007年的乐高10181,部件数目3428,原价199.99
    scrapePage(retX, retY, 'D:\PycharmProjects\linear regression/setHtml/lego10189.html', 2008, 5922, 299.99)  # 2008年的乐高10189,部件数目5922,原价299.99
    scrapePage(retX, retY, 'D:\PycharmProjects\linear regression/setHtml/lego10196.html', 2009, 3263, 249.99)  #2009年的乐高10196,部件数目3263,原价249.99

def useStandRegres():#使用简单的线性回归
    lgX=[];lgY=[]
    setDataCollect(lgX,lgY)
    data_num,features_num=np.shape(lgX)
    lgX1=np.mat(np.ones((data_num,features_num+1)))
    lgX1[:,1:5]=np.mat(lgX)
    ws=regression1.standRegres(lgX1,lgY)
    print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价'%(ws[0],ws[1],ws[2],ws[3],ws[4]))

def crossValidation(xArr,yArr,numVal=10):#交叉验证岭回归，numVal为交叉验证次数
    m=len(yArr)
    indexList=list(range(m))
    errorMat=np.zeros((numVal,30))
    for i in range(numVal):
        trainX=[];trainY=[]
        testX=[];testY=[]
        random.shuffle(indexList)
        for j in range(m):#随机生成测试集和训练集
            if j <m*0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat=regression4.ridgeTest(trainX,trainY)
        for k in range(30):
            matTestX=np.mat(testX);matTrainX=np.mat(trainX)
            meanTrain=np.mean(matTrainX,0)
            varTrain=np.var(matTrainX,0)
            matTestX=(matTestX-meanTrain)/varTrain
            yEst=matTestX*np.mat(wMat[k,:]).T+np.mean(trainY)
            errorMat[i,k]=regression3.rssError(yEst.T.A,np.array(testY))
    meanErrors=np.mean(errorMat,0)
    minMean=float(min(meanErrors))
    bestWeights=wMat[np.nonzero(meanErrors==minMean)]
    xMat=np.mat(xArr);yMat=np.mat(yArr).T
    meanX=np.mean(xMat,0);varX=np.var(xMat,0)
    unReg=bestWeights/varX
    print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价'%((-1*np.sum(np.multiply(meanX,unReg))+np.mean(yMat)),unReg[0,0],unReg[0,1],unReg[0,2],unReg[0,3]))

if __name__=='__main__':
    lgX=[];lgY=[]
    setDataCollect(lgX,lgY)
    print(regression4.ridgeTest(lgX,lgY))
    #crossValidation(lgX,lgY)#交叉验证岭回归
    #useStandRegres()        #简单线性回归



