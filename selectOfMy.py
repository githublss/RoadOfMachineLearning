#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by lss on 2019/2/25

import pandas as pd
import numpy as np
import datetime
import os

# 计算熵的函数
def ent(data):
    infor_entropy = pd.value_counts(data)/len(data)   # value_counts()，Return a Series containing counts of unique values.
    return sum(np.log2(infor_entropy) * infor_entropy * (-1))

# 计算信息增益
def gain(data,str1,str2):
    # 计算条件信息熵str1为条件，str2为分类结果
    e1 = data.groupby(str1).apply(lambda x: ent(x[str2]))  # 每个分支结点的信息熵
    p1 = pd.value_counts(data[str1]) / len(data[str1])     # 所占比例
    e2 = sum(e1*p1)
    return ent(data[str2])-e2

# 获取fpath文件夹下的type类型文件的列表
def findFileInFiles(fpath, type):
    files = os.listdir(fpath)
    F = []
    for file in files:
        if type in file:
            F.append(file)
    return F

# 将fpach文件夹下的.arff文件转换为csv文件
def arff_to_csv(fpath):
    fileNp=pd.read_csv(fpath,comment='@',header=None)
    fileNp=fileNp.replace('?', '0')     # 对数据集进行替换的替换规则
    fileNp = fileNp.replace('Y', '1')
    fileNp = fileNp.replace('N', '0')
    fpath=fpath.replace('.arff','.csv')
    print fpath
    fileNp.to_csv(fpath,index=False)
    print 'OK'


# 对fpath文件夹下的数据进行处理，从原始数据得到后面可以用到的数据
def dealData(fpath):
    arffFiles = findFileInFiles(fpath, '.arff')
    for file in arffFiles:
        arff_to_csv(fpath+file)
    csvFiles = findFileInFiles(fpath, 'allclass.csv')
    for csvFile in csvFiles:
        df = pd.read_csv(fpath + csvFile)
        df = df.ix[:, :-1]  # 前面是选择行，后面是选择列
        rankFeatureIndex = pd.read_csv('rankFeatureIndex.csv')  # 读取从weka中获得的特征排序的索引
        rankFeatureIndex['index'] = rankFeatureIndex['index'] - 1
        rankFeature = []  # 存放排好序的特征值
        for i in rankFeatureIndex['index']:
            rankFeature.append(df[str(i)])
        rankFeature = np.array(rankFeature).T  # 将list转化为numpy中的数组
        rankFeaturePd = pd.DataFrame(rankFeature, columns=rankFeatureIndex['index'])
        rankFeatureName = fpath + filter(str.isdigit, csvFile) + 'rankFeature.csv'
        rankFeaturePd.to_csv(rankFeatureName, index=False)  # 生成按照weka生成索引序列顺序的特征表
        print rankFeatureName + 'save OK!'

# 按照设计的特征选择算法，进行特征的选择
def getBestFeature(fpath):
    rankFiles = findFileInFiles(fpath,'rankFeature.csv')
    for rankFile in rankFiles:  # 遍历所有排好序的文件
        df = pd.read_csv(fpath+rankFile)  # 将排好序的特征值赋值给df
        featured = df.columns.values.tolist()  # 先将所有特征都添加到待选特征里面
        selectFeatureIndex = []  # 选取到的特征的索引
        deleteFeature = []  # 被剔除的特征的索引
        for feature1 in featured:   # 对文件中的每个特征进行遍历
            selectFeatureIndex.append(feature1)
            gainSum = 0  # 特征增益的和
            tempFeatureGain = []  # 存放feature1与其他特征间的信息增益列表
            feature2list = []  # 存放feature1与其他特征间的信息增益列表的索引
            for feature2 in featured:
                if feature2 in selectFeatureIndex:  # 为了不重复计算已经选择的属性之间的信息熵
                    continue
                feature2list.append(feature2)
                featureGain = gain(df, feature2, feature1)  # 计算出feature1与feature2之间的信息增益
                tempFeatureGain.append(featureGain)
                gainSum = gainSum + featureGain
                print feature1, feature2, featureGain
            tempFeatureGain = pd.Series(tempFeatureGain, index=feature2list)
            gainMean = tempFeatureGain.mean()
            print 'feature Gain list is:\n', tempFeatureGain
            print 'delete feature is:'
            for x in tempFeatureGain[tempFeatureGain > gainMean].index:  # 剔除的条件是信息增益是否大于平均值
                print x
                featured.remove(str(x))  # 将信息增益大于平均值的特征剔除掉
                deleteFeature.append(x)  # 将剔除的特征添加到剔除列表中
            print 'will select:', featured
            print 'selectd:', selectFeatureIndex
        pd.Series(selectFeatureIndex).to_csv(fpath+filter(str.isdigit,rankFile)+'selectFeatureIndex.csv', encoding='utf-8', index=False)

# 通过selectFeatureIndex.csv文件将特征值序列提取出来
def selectFeature(fpath):
    type = 'selectFeatureIndexCustom.csv'   # 抽取出来的特征索引顺序表
    toType = 'selectFeatureCustom.csv'      # 要保存成的文件名
    selectIndexs=findFileInFiles(fpath,type)
    for selectIndex in selectIndexs:
        rankFeaturePd = pd.read_csv(fpath+filter(str.isdigit,selectIndex)+'rankFeature.csv')
        selectFeatureIndex = pd.read_csv(fpath+filter(str.isdigit,selectIndex)+type, header=None, names=['index'])  # 将列名指定为index
        print selectFeatureIndex
        if 'Custom' in type:    # 自己定义的索引值从1开始的，所以减一
            selectFeatureIndex = selectFeatureIndex-1
        index = selectFeatureIndex['index']
        selectFeature = []  # 存放排好序的特征值
        for i in index:
            selectFeature.append(rankFeaturePd[str(i)])
        selectFeature = np.array(selectFeature).T  # 将list转化为numpy中的数组
        selectFeaturePd = pd.DataFrame(selectFeature, columns=selectFeatureIndex['index'])
        selectFeaturePd['target'] = pd.read_csv(fpath+'entry'+filter(str.isdigit,selectIndex)+'.weka.allclass.csv', encoding='utf-8')[
            '248']  # 将目标值添加到selectFeature中
        selectFeaturePd.to_csv(fpath+filter(str.isdigit,selectIndex)+toType, index=False)
        print 'ok'
def main():
    fpath = 'FeatureData/'
    # dealData(fpath)           # 将原始数据按照weka中获得特征索引整体进行排序
    # getBestFeature(fpath)     # 通过信息增益来选特征的索引值
    selectFeature(fpath)        # 通过上一步抽取出来的索引值，得到一个特征数据表


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    end = datetime.datetime.now()
    print '花费时间：'.decode('utf-8') , str(end - start)