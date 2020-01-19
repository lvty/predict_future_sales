# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('max_colwidth', 100)  # 设置value的显示长度为100，默认为50
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import learning_curve,cross_val_score
from sklearn.model_selection import train_test_split,StratifiedKFold,StratifiedShuffleSplit
from sklearn.metrics import classification_report
from sklearn.externals import joblib
le = preprocessing.LabelEncoder()
enc = preprocessing.OneHotEncoder()

test = pd.read_csv('competitive-data-science-predict-future-sales/test.csv',dtype={
        'ID':'int32',
        'shop_id':'int32',
        'item_id':'int32',
        })

train_monthly = pd.read_csv('train_monthly.csv')

train_monthly = train_monthly.query('item_cnt >= 0 and item_cnt <= 20 and item_price < 40000')
train_monthly.head().append(train_monthly.tail())

## 特征工程及线下验证
#时间序列问题在构造特征的时候就要包括到历史特征，还有时间窗特征，包括时间窗的
#sum,mean,median,std等，而线下验证集的划分则需要根据时间线来，不能用以往的交叉验证。

#1.首先是构造label特征，由于训练数据给的是13年1月到15年10月的数据，需要我们预测的是15年11月份的数据，
#所以我们将每组数据在时间线上向后平移，即1月份的label是2月份的销量，依此类推

train_monthly['item_cnt_month'] = train_monthly.sort_values(by='date_block_num').groupby(['shop_id','item_id'])['item_cnt'].shift(-1)

#商品的单位价格特征
train_monthly.eval('item_price_unit = item_price/item_cnt',inplace=True)
train_monthly['item_price_unit'].fillna(0,inplace=True)
#对于INF进行处理
train_monthly[np.isinf(train_monthly['item_price_unit'])] = 0


##计算每个产品的价格波动特征，包括最低价格和最高价格，以及价格增量和价格减量
gp_item_price = train_monthly.sort_values(by='date_block_num').groupby(['item_id'],as_index=False).agg({'item_price':[np.min,np.max]})
gp_item_price.columns = ['item_id','hist_min_item_price','hist_max_item_price']

train_monthly = pd.merge(train_monthly,gp_item_price,on='item_id',how='left')
train_monthly.eval('price_increase = item_price - hist_min_item_price',inplace=True)
train_monthly.eval('price_decrease = hist_max_item_price - item_price',inplace=True)

#2.构造时间窗特征，窗口大小设置为3，时间窗可以对数据起到平滑的效果，同时也包含了一定的历史信息。这里我们用时窗
#构造出min,max,mean以及std特征，并对缺失数据进行零填充。

f_min = lambda x:x.rolling(window=3,min_periods=1).min()
f_max = lambda x:x.rolling(window=3,min_periods=1).max()
f_mean = lambda x:x.rolling(window=3,min_periods=1).mean()
f_std = lambda x:x.rolling(window=3,min_periods=1).std()

funs = [f_min,f_max,f_mean,f_std]
fnames = ['min','max','mean','std']

for i in range(len(funs)):
    train_monthly[('item_cnt_%s' % fnames[i])] = train_monthly.sort_values(by='date_block_num').groupby(['shop_id','item_category_id','item_id'])['item_cnt'].apply(funs[i])

train_monthly['item_cnt_std'].fillna(0,inplace=True)

#构造滞后历史特征，将历史三个月的数据平移过来
lag_list = [1,2,3]
for lag in lag_list:
    ft_name = ('item_cnt_shifted%s' % lag)
    train_monthly[ft_name] = train_monthly.sort_values(by='date_block_num').groupby(['shop_id','item_category_id','item_id'])['item_cnt'].shift(lag)
    train_monthly[ft_name].fillna(0,inplace=True)

#构造销量变化特征，通过计算滞后历史特征的变化量来得出
train_monthly['item_trend'] = train_monthly['item_cnt']

for lag in lag_list:
    ft_name = ('item_cnt_shifted%s' % lag)
    train_monthly['item_trend'] -= train_monthly[ft_name]

train_monthly['item_trend'] /= len(lag_list) + 1


#先划分数据集和验证集，这里由于我们使用了滞后平移操作以及时间窗的计算，所以我们丢弃前三个月的数据，
#同时将date_block_num介于27到33之间的数据划分为线下的验证集
train_set = train_monthly.query('date_block_num >= 3 and date_block_num < 28').copy()
validation_set = train_monthly.query('date_block_num >= 28 and date_block_num < 33').copy()
test_set = train_monthly.query('date_block_num == 33').copy()

train_set.dropna(subset=['item_cnt_month'],inplace=True)
validation_set.dropna(subset=['item_cnt_month'],inplace=True)

train_set.dropna(inplace=True)
validation_set.dropna(inplace=True)

#分别对商店，商品，年和月构造销量的均值特征
gp_shop_mean = train_set.groupby(['shop_id']).agg({'item_cnt_month':['mean']})
gp_shop_mean.columns=['shop_mean']
gp_shop_mean.reset_index(inplace=True)

gp_item_mean = train_set.groupby(['item_id']).agg({'item_cnt_month':['mean']})
gp_item_mean.columns=['item_mean']
gp_item_mean.reset_index(inplace=True)

gp_shop_item_mean = train_set.groupby(['shop_id','item_id']).agg({'item_cnt_month':['mean']})
gp_shop_item_mean.columns=['shop_item_mean']
gp_shop_item_mean.reset_index(inplace=True)

gp_year_mean = train_set.groupby(['year']).agg({'item_cnt_month':['mean']})
gp_year_mean.columns=['year_mean']
gp_year_mean.reset_index(inplace=True)

gp_month_mean = train_set.groupby(['month']).agg({'item_cnt_month':['mean']})
gp_month_mean.columns=['month_mean']
gp_month_mean.reset_index(inplace=True)

train_set = pd.merge(train_set,gp_shop_mean,on='shop_id',how='left')
train_set = pd.merge(train_set,gp_item_mean,on='item_id',how='left')
train_set = pd.merge(train_set,gp_shop_item_mean,on=['shop_id','item_id'],how='left')
train_set = pd.merge(train_set,gp_year_mean,on='year',how='left')
train_set = pd.merge(train_set,gp_month_mean,on='month',how='left')

## todo???????????
validation_set = pd.merge(validation_set,gp_shop_mean,on='shop_id',how='left')
validation_set = pd.merge(validation_set,gp_item_mean,on='item_id',how='left')
validation_set = pd.merge(validation_set,gp_shop_item_mean,on=['shop_id','item_id'],how='left')
validation_set = pd.merge(validation_set,gp_year_mean,on='year',how='left')
validation_set = pd.merge(validation_set,gp_month_mean,on='month',how='left')

#分离出训练集和验证集的X和Y
X_train = train_set.drop(['item_cnt_month','date_block_num'],axis=1)
Y_train = train_set['item_cnt_month'].astype(int)

X_validation = validation_set.drop(['item_cnt_month','date_block_num'],axis=1)
Y_validation = validation_set['item_cnt_month'].astype(int)

#类型转换
fs = ['shop_id','item_id','year','month']

X_train[fs] = X_train[fs].astype('int32')
X_validation[fs] = X_validation[fs].astype('int32')

#对测试集进行缺失特征的填充，填充规则为最近的一个月的特征，所以这里构造了一个lastest_records，
#latest_records为每个shop_id,item_id组合的最新的特征记录。
last_records = pd.concat([train_set,validation_set]).drop_duplicates(subset=['shop_id','item_id'],keep='last')

X_test = pd.merge(test,last_records,on=['shop_id','item_id'],how='left',suffixes=['','_'])
X_test.head().append(X_test.tail())
X_test['year']=2015
X_test['month']=9
X_test.drop('item_cnt_month',axis=1,inplace=True)
X_test[fs] = X_test[fs].astype('int32')
X_test = X_test[X_train.columns]


#对缺失数据按照shop_id进行各列的中位数填充

sets = [X_train,X_validation,X_test]

for ds in sets:
    for shop_id in ds['shop_id'].unique():
        for col in ds.columns:
            shop_median = ds[(ds['shop_id'] == shop_id)][col].median()
            ds.loc[(ds[col].isnull()) & (ds['shop_id'] == shop_id),col] = shop_median

X_test.fillna(X_test.mean(),inplace=True)

X_train.drop(['item_category_id'],axis=1,inplace=True)
X_validation.drop(['item_category_id'],axis=1,inplace=True)
X_test.drop(['item_category_id'],axis=1,inplace=True)


X_train.to_csv('train.csv',index=False)
X_validation.to_csv('validation.csv',index=False)
X_validation.to_csv('test.csv',index=False)



#stacking的算法步骤，对于每个模型，可以获取到valid_predict和test_predict,将原来Y_valid作为label，
#valid_pridict和test_predict作为features和test，堆叠在一起;第二层，选择模型，并训练，对测试集预测；
















