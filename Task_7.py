import numpy as np 
import pandas as pd 
from pandas import DataFrame, Series

# 12.1.1背景与目标
"""
values = Series(['apple', 'orange', 'apple', 'apple']*2)
print(values)
print(pd.unique(values))
print(pd.value_counts(values))

values = Series([0, 1, 0, 0]*2)
dim = Series(['apple', 'orange'])
print(values)
print(dim)
print(dim.take(values))

# 12.1.2pandas中的categorical类型
fruits = ['apple', 'orange', 'apple', 'apple'] * 2
N = len(fruits)
df = DataFrame({'fruit' : fruits,
                'basket_id' : np.arange(N),
                'count' : np.random.randint(3, 15, size=N),
                'weight': np.random.uniform(0, 4, size=N),
                }, 
                columns=['basket_id', 'fruit', 'count', 'weight'])

print(df)
print(df['fruit'])
fruit_cat = df['fruit'].astype('category')
print(fruit_cat)

c = fruit_cat.values
print(type(c))
print(c.categories)
print(c.codes)

df['fruit'] = df['fruit'].astype('category')
print(df.fruit)

my_categories = pd.Categorical(['foo', 'bar', 'baz', 'foo', 'bar'])
print(my_categories)

categories = ['foo', 'bar', 'baz']
codes = [0, 1, 2, 0, 0, 1]
my_cats_2 = pd.Categorical.from_codes(codes, categories)
print(my_cats_2)

ordered_cat = pd.Categorical.from_codes(codes, categories, ordered=True)
print(ordered_cat)
print(my_cats_2.as_ordered())

# 12.1.3使用Categorical对象进行计算
np.random.seed(12345)
draws = np.random.randn(1000)
print(draws)
print(len(draws))

# 计算上面数据的四分位分箱，并提取一些统计值
bins = pd.qcut(draws, 4)
print(bins)
bins = pd.qcut(draws, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
print(bins)
print(bins.codes[:])

bins = Series(bins, name='quartile')
results = (Series(draws)
           .groupby(bins)
           .agg(['count', 'min', 'max'])
           .reset_index())

print(results)
"""

# 12.1.3.1使用分类获得更高性能
N = 10000000
draws = Series(np.random.randn(N))
labels = Series(['foo', 'bar', 'baz', 'qux'] * (N // 4))
categories = labels.astype('category')
print(labels.memory_usage())
print("hello world")
print(categories.memory_usage())

# 12.1.4分类方法
s = Series(['a', 'b', 'c', 'd'] * 2)
cat_s = s.astype('category')
print(cat_s)
print(cat_s.cat.codes)

actual_categories = ['a','b', 'c', 'd', 'e']
cat_s2 = cat_s.cat.set_categories(actual_categories)
print(cat_s2)
print(cat_s.value_counts())

cat_s3 = cat_s[cat_s.isin(['a', 'b'])]
print(cat_s3)
print(cat_s3.cat.remove_unused_categories())

# 12.1.4.1创建用于建模的虚拟变量
cat_s = Series(['a', 'b', 'c', 'd'] * 2, dtype='category')
print(pd.get_dummies(cat_s))
# get_dummies函数将一堆一维的分类数据转换为一个包含虚拟变量的DataFrame

# 12.2高阶GroupBy应用
# 12.2.1分组转换和“展开”GroupBy
df = DataFrame({'key' : ['a', 'b', 'c']*4,
                'value': np.arange(12.)})

print(df)
g = df.groupby('key').value
print(g.mean())
print(g.transform(lambda x: x.mean()))
print(g.transform('mean'))
print(g.transform(lambda x: x*2))
print(g.transform(lambda x: x.rank(ascending=False)))

def normalize(x):
    return(x - x.mean() / x.std())

print(g.transform(normalize))
print(g.apply(normalize))
print(g.transform('mean'))

normalized = (df['value'] - g.transform('mean')) / g.transform('std')
print(normalized)

# 12.2.2分组的时间重新采样
N = 15
times = pd.date_range('2017-05-20 00:00', freq='1min', periods=N)
df = DataFrame({'time' : times,
                'value': np.arange(N)})

print(df)
print(df.set_index('time').resample('5min').count())
df2 = DataFrame({'time' : times.repeat(3),
                 'key'  : np.tile(['a', 'b', 'c'], N),
                 'value': np.arange(N * 3.)})
print(df2)
time_key = pd.TimeGrouper('5min')
resampled = (df2.set_index('time')
             .groupby(['key', time_key])
             .sum())

print(resampled)
print(resampled.reset_index())

# 12.3方法链技术

# 12.3.1pipe方法