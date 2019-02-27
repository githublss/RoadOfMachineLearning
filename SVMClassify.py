#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by lss on 2019/2/24

# 通过使用sklearn中的集成算法中的SVM，来对前面筛选出来的特征值进行分类
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score,classification_report
from selectOfMy import findFileInFiles
import datetime

# 将目标值从字符转换为浮点型数据
def target_type(s):
    it = {'WWW':0, 'SERVICES':1, 'P2P':2, 'MULTIMEDIA':3, 'MAIL':4, 'INTERACTIVE':5, 'GAMES':6, 'FTP-PASV':7,
          'FTP-DATA':8, 'FTP-CONTROL':9, 'DATABASE':10, 'ATTACK':11}
    return it[s]

def main():
    fpath = 'FeatureData/'
    useData = 'selectFeature.csv'   # 使用的数据集
    files = findFileInFiles(fpath,useData)[:6]  # 只计算前六个数据集
    print files
    for file in files:
        start = datetime.datetime.now()
        path =file
        temp = pd.read_csv(fpath+path)
        # print temp.shape[1]
        target=temp.shape[1]-1   # 分类目标值所在的列号
        data = np.loadtxt(fpath+path,dtype=float,delimiter=',',skiprows=1,converters={target:target_type})   # 将第66列的分类结果用数字来表示,忽略第一行的列名称
        x,y = np.split(data,(target,),axis=1)   # axis=1表示在水平方向上进行分割
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)   # 把样本占比设为0.7

        clf = svm.SVC(C=16, kernel='rbf', gamma=0.065, decision_function_shape='ovr')
        clf.fit(x_train, y_train.ravel())

        # print clf.score(x_train,y_train)    #精度
        # y_hat = clf.predict(x_train)
        # print accuracy_score(y_hat, y_train, '训练集')
        # print clf.score(x_test, y_test)
        targetName = ['WWW', 'SERVICES', 'P2P', 'MULTIMEDIA', 'MAIL', 'INTERACTIVE', 'GAMES', 'FTP-PASV',
          'FTP-DATA', 'FTP-CONTROL', 'DATABASE', 'ATTACK']
        y_pred = clf.predict(x_test)
        print filter(str.isdigit,file),':'
        print '选取特征数量：'.decode('utf-8'),temp.shape[1]
        print accuracy_score(y_pred, y_test)
        print classification_report(y_test,y_pred,digits=4)
        end = datetime.datetime.now()
        print filter(str.isdigit,file)+'花费时间：'.decode('utf-8'), str(end - start)
        # print data

if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    end = datetime.datetime.now()
    print '花费时间：'.decode('utf-8'), str(end - start)