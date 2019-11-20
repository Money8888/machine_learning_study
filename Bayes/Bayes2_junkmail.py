import re
import Bayes1

def textParse(bigString):                                #将文本内容的大字符串转换成字符串列表
    '''regEx=re.compile('\\W*')
    listOfTokens = regEx.split(bigString)'''
    #listOfTokens=re.split(r'\w*', bigString)
    listOfTokens=bigString.split()
    #return [tok.lower() for tok in listOfTokens if len(tok) > 2]
    return [tok.lower() for tok in listOfTokens if len(tok) > 1]  #删除单个字母，其余字母小写
    #return listOfTokens
if __name__=='__main__':                                 #spam文件下的txt文件为垃圾邮件
    docList=[];classList=[]
    for i in range(1,26):
        #print(open('spam/%d.txt'%i,'r').read())
        wordList = textParse(open('spam/%d.txt' % i, 'r').read())
        #print(wordlist)
        docList.append(wordList)
        #print(doclist)
        classList.append(1)
        #print(open('ham/%d.txt' % 1, 'r').read())
        wordList = textParse(open('ham/%d.txt' % i, 'r').read())
        docList.append(wordList)
        classList.append(0)
        vocabList = Bayes1.
    print(vocabList)
'''if __name__=='__main__':
    bigString='hjwedxjksdcnwkjdcjk dwjekgcbjwe dwbedgwjd dwukggyqdhqdnjwv  dbkuqedckquh'
    print(textParse(bigString))'''

