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

item_categories = pd.read_csv('competitive-data-science-predict-future-sales/item_categories.csv',dtype={
        'item_category_name':'str',
        'item_category_id':'int32'
        })

items = pd.read_csv('competitive-data-science-predict-future-sales/items.csv',dtype={
        'item_name':'str',
        'item_id':'int32',
        'item_category_id':'int32'
        })


shops = pd.read_csv('competitive-data-science-predict-future-sales/shops.csv',dtype={
        'shop_name':'str',
        'shop_id':'int32'
        })


#parse_dates : boolean or list of ints or names or list of lists or dict, default False
#boolean. True -> 解析索引
#list of ints or names. e.g. If [1, 2, 3] -> 解析1,2,3列的值作为独立的日期列；
#list of lists. e.g. If [[1, 3]] -> 合并1,3列作为一个日期列使用
#dict, e.g. {‘foo’ : [1, 3]} -> 将1,3列合并，并给合并后的列起名为"foo"
#dtype : Type name or dict of column -> type, default None
#每列数据的数据类型。例如 {‘a’: np.float64, ‘b’: np.int32}

sales = pd.read_csv('competitive-data-science-predict-future-sales/sales_train.csv',
        parse_dates=['date'],dtype={'date': 'str','date_block_num': 'int32','shop_id': 'int32', 'item_id': 'int32','item_price': 'float32','item_cnt_day': 'int32'
        })

#rsuffix是用来标记重复的column列，合并之后会被drop掉
train = sales.join(items,on='item_id',rsuffix='_').join(shops,on='item_id',rsuffix='_').join(item_categories,on='item_category_id',rsuffix='_').drop(['item_id_','item_id_','item_category_id_'],axis=1)

train.head().append(train.tail())

#train.shape   #(2935849, 11)

test.head().append(test.tail())

#一共有2935849个样本，10列原始特征列，记录的销量日期从2013年1月到2015年10月，而这里需要我们预测的是2015年11月的销量信息

#trick

#这里实际上是个tricks，从测试集可以看出，测试集只有shop_id和item_id，且行数少于训练集行数，考虑模型只训练测试集所包含的（shop_id,item_id）对，其他的匹配对不予以考虑，在数据竞赛中这个tricks可以提升不少分数;

#leakages

test_shop_ids = test.shop_id.unique()
test_item_ids = test.item_id.unique()

lk_train = train[train.shop_id.isin(test_shop_ids)]
lk_train = lk_train[lk_train.item_id.isin(test_item_ids)]

train_monthly = lk_train[['date','date_block_num','shop_id','item_category_id','item_id','item_price','item_cnt_day']]


##计算销量和价格的总值和均值
train_monthly=train_monthly.sort_values(by='date').groupby(['date_block_num','shop_id','item_category_id','item_id'],as_index=False)
train_monthly = train_monthly.agg({
        'item_price':['sum','mean'],
        'item_cnt_day':['sum','mean','count']
        })
##重命名
train_monthly.columns = ['date_block_num','shop_id','item_category_id','item_id','item_price','mean_item_price','item_cnt','mean_item_cnt','transactions']

train_monthly.head(5).append(train_monthly.tail(5))

#考虑到测试集可能会有不同的商店和商品的组合，这里我们对训练数据按照shop_id和item_id的组合进行扩充，缺失数据进行零填充，同时构造出具体的年，月信息

shop_ids = train_monthly['shop_id'].unique()
item_ids = train_monthly['item_id'].unique()

empty_df = []
for i in range(34):
    for shop in shop_ids:
        for item in item_ids:
            empty_df.append([i,shop,item])

empty_df = pd.DataFrame(empty_df,columns=['date_block_num','shop_id','item_id'])

train_monthly = pd.merge(empty_df,
                         train_monthly,
                         on=['date_block_num','shop_id','item_id'],
                         how='left'
                         )
train_monthly.fillna(0,inplace=True)

train_monthly['year'] = train_monthly['date_block_num'].apply(lambda x:int((x/12) + 2013))
train_monthly['month'] = train_monthly['date_block_num'].apply(lambda x:(x%12))


train_monthly.shape ## (6734448, 11)

train_monthly.head()

train_monthly.to_csv('train_monthly.csv',index=False)

#EDA
#item_cnt为每天产品的总销量
gp_month_mean = train_monthly.groupby(['month'],as_index=False)['item_cnt'].mean()
gp_month_sum = train_monthly.groupby(['month'],as_index=False)['item_cnt'].sum()

gp_category_mean = train_monthly.groupby(['item_category_id'],as_index=False)['item_cnt'].mean()
gp_category_sum = train_monthly.groupby(['item_category_id'],as_index=False)['item_cnt'].sum()

gp_shop_mean = train_monthly.groupby(['shop_id'],as_index=False)['item_cnt'].mean()
gp_shop_sum = train_monthly.groupby(['shop_id'],as_index=False)['item_cnt'].sum()

#分组查看各组和销量之间的关系曲线
f,ax = plt.subplots(2,1,figsize=(22,10),sharex=True)
sns.lineplot(x='month',y='item_cnt',data=gp_month_mean,ax=ax[0]).set_title('Monthly mean')
sns.lineplot(x='month',y='item_cnt',data=gp_month_sum,ax=ax[1]).set_title('Monthly sum')
plt.show()

#可以看出，时间上下半年的销量是上升的

f,ax = plt.subplots(2,1,figsize=(22,10),sharex=True)
sns.barplot(x='item_category_id',y='item_cnt',data=gp_category_mean,ax=ax[0],palette='rocket').set_title('Monthly mean')
sns.barplot(x='item_category_id',y='item_cnt',data=gp_category_sum,ax=ax[1],palette='rocket').set_title('Monthly sum')
plt.show()

#发现只有部分category对销量是有突出的贡献

f,ax = plt.subplots(2,1,figsize=(22,10),sharex=True)
sns.barplot(x='shop_id',y='item_cnt',data=gp_shop_mean,ax=ax[0],palette='rocket').set_title('Monthly mean')
sns.barplot(x='shop_id',y='item_cnt',data=gp_shop_sum,ax=ax[1],palette='rocket').set_title('Monthly sum')
plt.show()

#有三个突出的商店销量比较高，考虑到可能是大商场或者比较出名的零售店，可以用来后续的特征工程构造


#通过散点关联和箱形图查看异常值

#jointplot是画两个变量或者单变量的图像，是对JointGrid类的实现
#x,y为DataFrame中的列名或者是两组数据，data指向dataframe ,kind是你想要画图的类型
#stat_func 用于计算统计量关系的函数
#kind 图形的类型scatter,reg,resid,kde,hex
#如果不显示r值(pearsonr),可以在参数中添加stat_func=sci.pearsonr,有就不用添加了
#sns.jointplot(x=df['A'], y=df['B'], #设置xy轴，显示columns名称
#              data = df,  #设置数据
#              color = 'b', #设置颜色
#              s = 50, edgecolor = 'w', linewidth = 1,#设置散点大小、边缘颜色及宽度(只针对scatter)
#              stat_func=sci.pearsonr,
#              kind = 'scatter',#设置类型：'scatter','reg','resid','kde','hex'
#              #stat_func=<function pearsonr>,
#              space = 0.1, #设置散点图和布局图的间距
#              size = 8, #图表大小(自动调整为正方形))
#              ratio = 5, #散点图与布局图高度比，整型
#              marginal_kws = dict(bins=15, rug =True), #设置柱状图箱数，是否设置rug
#)

sns.jointplot(x='item_cnt',y='item_price',data=train_monthly,height=8)
plt.show()

sns.jointplot(x='item_cnt',y='transactions',data=train_monthly,height=8)
plt.show()

plt.subplots(figsize=(22,8))
sns.boxplot(train_monthly['item_cnt'])
plt.show()

#认为销量不在[0,20]，售价超过40000的为异常值，剔除异常值即可。预处理之后的训练集如下所示
train_monthly = train_monthly.query('item_cnt >= 0 and item_cnt <= 20 and item_price < 40000')
train_monthly.head().append(train_monthly.tail())



