import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import Adaboost2

if __name__=='__main__':
    dataArr,classLabels=Adaboost2.loadDataSet('horseColicTraining2.txt')
    testArr, testLabelArr = Adaboost2.loadDataSet('horseColicTest2.txt')
    bdt=AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),algorithm="SAMME",n_estimators=10)
    print(bdt)
    bdt.fit(dataArr,classLabels)
    predictions=bdt.predict(dataArr)
    errArr = np.mat(np.ones((len(dataArr), 1)))
    print('训练集的错误率：%.3f%%'%(float(errArr[predictions!=classLabels].sum()*100/len(dataArr))))
    predictions=bdt.predict(testArr)
    errArr = np.mat(np.ones((len(testArr), 1)))
    print('测试集的错误率：%.3f%%'%(float(errArr[predictions!=testLabelArr].sum()*100/len(testArr))))
