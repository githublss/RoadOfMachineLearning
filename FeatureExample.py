#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by lss on 2019/2/18

import pandas as pd
import numpy as np

def main():
    data1 = pd.DataFrame({'天气':['晴','晴','阴','雨','雨','雨','阴','晴','晴','雨','晴','阴','阴','雨'],
                         '温度':['高','高','高','低','低','低','低','低','低','低','低','低','高','低'],
                         '湿度':['高','低','高','高','高','低','低','高','低','高','低','高','低','高'],
                         '起风':[False,True,False,False,False,True,True,False,False,False,True,True,False,True],
                         '打球':['NO','NO','YES','YES','YES','NO','YES','NO','YES','YES','YES','YES','YES','NO']})
    print data1[['天气','温度','湿度','起风','打球']]
    # 计算法熵
    def ent(data):
        infor_entropy = pd.value_counts(data)/len(data)   # value_counts()，Return a Series containing counts of unique values.
        return sum(np.log2(infor_entropy) * infor_entropy * (-1))
    def gain(data,str1,str2):
        # 计算条件信息熵
        e1 = data1.groupby(str1).apply(lambda x: ent(x[str2]))  # 每个分支结点的信息熵
        p1 = pd.value_counts(data1[str1]) / len(data1[str1])     # 所占比例
        e2 = sum(e1*p1)
        print ent(data[str2])-e2
    gain(data1,'天气', '打球')
    print data1[['天气','湿度']]
    print data1.columns.values.tolist()[1]
    g = [1,9,3,4,5,6,7,8,2,3,4,5,6,6,6]

    g = pd.Series(g)
    for i in g:
        print g
        for x in g[g>g.mean()].index:
            g.pop(x)
        # g = g.drop(g[g>g.mean()].index)
        # if len(g)<=2:
        #     break
if __name__ == '__main__':
    main()