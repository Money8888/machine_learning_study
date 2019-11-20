import numpy as np
from bs4 import BeautifulSoup
import random
import regression6
from sklearn import linear_model

def usesklearn():
    reg=linear_model.Ridge(alpha=0.5)
    lgX=[];lgY=[]
    regression6.setDataCollect(lgX,lgY)
    reg.fit(lgX,lgY)
    print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价'%(reg.intercept_,reg.coef_[0],reg.coef_[1],reg.coef_[2],reg.coef_[3]))

if __name__=='__main__':
    usesklearn()