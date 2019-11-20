import numpy as np
import operator
from os import listdir
from sklearn.svm import SVC
import SVM4

def handwritingClassTest():
    hwLabels=[]
    trainingFileList=listdir('trainingDigits')
    m=len(trainingFileList)
    trainingMat=np.zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        classNumber=int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i,:]=SVM4.img2vector('trainingDigits/%s'%(fileNameStr))
    clf=SVC(C=200,kernel='rbf')
    clf.fit(trainingMat,hwLabels)
    testFileList=listdir('testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        classNumber=int(fileNameStr.split('_')[0])
        vectorUnderTest=SVM4.img2vector('testDigits/%s'%(fileNameStr))
        classifierResult=clf.predict(vectorUnderTest)
        print("分类返回结果为%d\t真实结果为%d"%(classifierResult,classNumber))
        if(classifierResult!=classNumber):
            errorCount+=1.0
    print("总共错了%d个数据\n错误率为%f%%"%(errorCount,errorCount*100/mTest))

if __name__=='__main__':
    handwritingClassTest()