'''from sklearn import tree

if __name__=='__main__':
    fr=open('lenses.txt')
    lenses=[inst.strip().split('\t') for inst in fr.readlines()]
    print(lenses)
    lensesLabels=['age','prescript','astigmatic','tearRate']
    clf=tree.DecisionTreeClassifier()
    lenses=clf.fit(lenses,lensesLabels)'''#错误代码，因为fit不能接受string类型的数据

import pandas as pd     #采用pandas数据类型
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.externals.six import StringIO
from sklearn import tree
import pydotplus


if __name__=='__main__':
    with open('lenses.txt','r') as fr:           #with防止出现文件异常
        lenses=[inst.strip().split('\t') for inst in fr.readlines()]
        #print(lenses)
    lenses_target=[]
    for each in lenses:
        lenses_target.append(each[-1])

    lensesLabels=['age','prescript','astigmatic','tearRate'] #各个特征标签
    lenses_list=[]
    lenses_dict={}
    for each_labels in lensesLabels:
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_labels)])
        lenses_dict[each_labels]=lenses_list
        lenses_list=[]
    #print(lenses_dict)
    lenses_pd=pd.DataFrame(lenses_dict)
    #print(lenses_pd)
    le=LabelEncoder()                    #创建LabelEncoder对象,将字符串数字化
    for col in lenses_pd.columns:
        lenses_pd[col]=le.fit_transform(lenses_pd[col]) #每一列数字化（序列化）
    #print(lenses_pd)

    clf = tree.DecisionTreeClassifier(max_depth=4)  #最大深度指除了根节点所在层数的总层数，即此时树有5层
    clf = clf.fit(lenses_pd.values.tolist(),lenses_target)  #构建决策树
    #print(clf)
    dot_data = StringIO()
    tree.export_graphviz(clf,out_file=dot_data,feature_names=lenses_pd.keys(),class_names=clf.classes_,filled=True,rounded=True,special_characters=True)
    graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("tree.pdf")
    print(clf.predict([[1,1,1,0]]))  #预测，其中的数据为所给的特征标签数字化来的，调用预测时，需将字符串转化为数字



