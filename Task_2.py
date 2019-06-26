import pandas as pd 
import numpy as np 
from pandas import DataFrame, Series

# 5.1
"""
# Series
obj = Series([4, 7, -5, 3])
print(obj, obj.values, obj.index)

obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
print(obj2)
print(obj2.index)
print(obj2[obj2 > 0])
print(obj * 2)
print(np.exp(obj2))
print('b' in obj2)

sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(sdata)
print(obj3)
states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(sdata, index=states)
print("obj4", "\n",obj4)
print(pd.isnull(obj4))
print(pd.notnull(obj4))
print(obj3 + obj4)
obj4.name = 'population'
obj4.index.name = 'state'
print(obj4)
obj.index = ['Bob', 'steve', 'Jeff', 'Ryan']
obj.name = 'Name'
obj.index.name = 'Name'
print(obj)
"""

# 5.1.2DataFrame
"""
data = {'state' : ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year'  : [2000, 2001, 2002, 2001, 2002, 2003],
        'pop'   : [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = DataFrame(data)
print(DataFrame(data, columns=['year', 'state', 'pop']))

frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'], index=['one', 'two', 'three', 'four', 'five', 'six'])
print(frame2)
print(frame2.columns)
print(frame2['state'])
print(frame2.year)
print(frame2.loc['three'])
frame2['debt'] = 16.5
print(frame2)
frame2['debt'] = np.arange(6.)
print(frame2)

val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['dent'] = val
print(frame2)
frame2['eastern'] = frame2.state == 'Ohio'
print(frame2)
del frame2['eastern'] 
print(frame2.columns)
pop = {'Nevada' : {2001: 2.4, 2002: 2.9},
       'Ohio'   : {2000: 1.5, 2002: 3.6, 2001: 3.6}}
frame3 = DataFrame(pop)
print(frame3)

print(frame3.T)

pdata = {'Ohio'  : frame3['Ohio'][:-1],
         'Nevada': frame3['Nevada'][:2]}
print(DataFrame(pdata))

frame3.index.name = 'year'; frame3.columns.name = 'state'
print(frame3)
print(frame3.values)
print(frame2.values)
"""

"""
# 5.1.3索引对象
obj = Series(range(3), index=['a', 'b', 'c'])
index = obj.index
print(index)
print(index[1:])

labels = pd.Index(np.arange(3))
print(labels)

obj2 = Series([1.5, -2.5, 0], index=labels)
print(obj2)

dup_labels = pd.Index(['foo', 'foo', 'bar', 'bar'])
print(dup_labels)
"""

# 5.2.1重建索引
"""
obj = Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
print(obj)

obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
print(obj2)

obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
print(obj3)
obj3.reindex(range(6), method='ffill')
print(obj3)

frame = DataFrame(np.arange(9).reshape((3, 3)), 
                  index=['a', 'c', 'd'],
                  columns=['Ohio', 'Texas', 'California'])

print(frame)

frame2 = frame.reindex(['a', 'b', 'c', 'd'])
print(frame2)

states = ['Texas', 'Utah', 'California']
frame.reindex(columns=states)

print(frame.tz_loc[['a', 'b', 'c', 'd'], states])
"""

# 5.2.2轴向上删除条目
"""
obj = Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
print(obj)

new_obj = obj.drop('c')
print(new_obj)
print(obj.drop(['d', 'c']))

data = DataFrame(np.arange(16).reshape((4, 4)),
                 index=['Ohio', 'Colorado', 'Utah','New York'],
                 columns=['one', 'two', 'three', 'four'])

print(data)
print(data.drop(['Colorado', 'Ohio']))
print(data.drop('two', axis=1))
print(data.drop(['two', 'four'], axis='columns'))
obj.drop('c', inplace=True)
print(obj)
"""

# 5.2.3索引、选择与过滤
"""
obj = Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
print(obj)

print(obj['b'])
print(obj[2:4])
print(obj[['b', 'a', 'd']])
print(obj[[1, 3]])
print(obj[obj<2])

print(obj['b':'c'])
obj['b':'c'] = 5
print(obj)

data = DataFrame(np.arange(16).reshape((4, 4)),
                 index=['Ohio', 'Colorado', 'Utah','New York'],
                 columns=['one', 'two', 'three', 'four'])
print(data)
print(data['two'])
print(data[['three', 'one']])
print(data[:2])
print(data[data['three']>5])
print(data < 5)
data[data < 5] = 0
print(data)

# 5.2.3.1使用loc和iloc选择数据
print(data.loc['Colorado', ['two', 'three']])
print(data.iloc[2, [3, 0, 1]])
print(data.iloc[[1,2], [3, 0, 1]])
print(data)
print(data.loc[: 'Utah', 'two'])
print(data.iloc[:, :3][data.three > 5])

# 5.2.4整数索引
ser = Series(np.arange(3.))
print(ser)
ser2 = Series(np.arange(3.), index=['a', 'b', 'c'])
print(ser2[-1])
print(ser[:1])
print(ser.loc[:1])
print(ser.iloc[:1])

# 5.2.5算术和数据对齐
s1 = Series([7.2, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])
print(s1)
print(s2)
print(s1 + s2)

df1 = DataFrame(np.arange(9.).reshape((3, 3)),
                columns=list('bcd'),
                index=['Ohio', 'Texas', 'Colorado'])

df2 = DataFrame(np.arange(12.).reshape((4, 3)),
                columns=list('bde'),
                index=['Utah', 'Ohio', 'Texas', 'Oregon'])
print(df1)
print(df2)

print(df1 + df2)

df1 = DataFrame({'A': [1, 2]})
df2 = DataFrame({'B': [3, 4]})
print(df1)
print(df2)
print(df1 - df2)
"""

# 5.2.5.1使用填充值的算术方法
"""
df1 = DataFrame(np.arange(12.).reshape((3, 4)),
                columns=list('abcd'),)
df2 = DataFrame(np.arange(20.).reshape(4, 5),
                columns=list('abcde'))

df2.loc[1, 'b'] = np.nan
print(df2)

print(df1 + df2)
print(df1.add(df2, fill_value=0))
print(1 / df1)
print(df1.rdiv(1))
df1.reindex(columns=df2.columns, fill_value=0)
print(df1)

# 5.2.5.2DataFrame和Series间的操作
arr = np.arange(12.).reshape((3, 4))
print(arr)
print(arr[0])
arr - arr[0]

frame = DataFrame(np.arange(12.).reshape((4, 3)),
                  columns=list('bde'),
                  index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series = frame.iloc[0]
print(frame)
print(series)
print(frame - series)
series2 = Series(range(3), index=['b', 'e', 'f'])
print(frame + series2)

series3 = frame['d']
print(frame)
print(series3)

print(frame.sub(series3, axis='index'))
"""

# 5.2.6函数应用和映射
"""
frame = DataFrame(np.random.randn(12).reshape((4, 3)),
                  columns=list('bde'),
                  index=['Utah', 'Ohio', 'Texas', 'Oregon'])
print(frame)
print(np.abs(frame))

f = lambda x : x.max() - x.min()
print(frame.apply(f))

print(frame.apply(f, axis='columns'))

format = lambda x: '%.2f' %x
print(frame.applymap(format))
"""

# 5.2.7排名和排序
"""
obj = Series(range(4), index=['d', 'a', 'b', 'c'])
print(obj.sort_index())

frame = DataFrame(np.arange(8).reshape((2, 4)),
                  index=['three', 'one'],
                  columns=['d', 'a', 'b', 'c'])

print(frame.sort_index(axis=1, ascending=False))

obj = Series([4, 7, -3, 1])
print(obj.sort_values())
obj = Series([4, np.nan, np.nan, -3, 2])
print(obj.sort_values())

frame = DataFrame({'b' : [4, 7, -3, 2],
                   'a' : [0, 1, 0, 1]})
print(frame)
print(frame.sort_values(by='b'))
print(frame.sort_values(by=['a', 'b']))

obj = Series([7, -5, 4, 2, 0, 4])
print(obj.rank())
print(obj.rank(method='first'))
print(obj.rank(ascending=False, method='max'))
"""


"""
frame = DataFrame({'b' : [4.3, 7, -3, 2],
                   'a' : [0, 1, 0, 1],
                   'c' : [-2, 5, 8, -2.5]})
print(frame)
print(frame.rank(axis='columns', ascending=False))

# 5.2.8含有重复值标签的轴索引
obj = Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
print(obj)
print(obj.index.is_unique)

print(obj['a'])
print(obj['c'])

df = DataFrame(np.random.randn(4, 3), index=['a', 'a', 'b', 'b'])
print(df)
print(df.loc['b'])
"""

# 5.3描述性统计的概述和计算
"""
df = DataFrame([[1.4, np.nan],
                [7.1, -4.5],
                [np.nan, np.nan],
                [0.75, -1.3]], 
                index = ['a', 'b', 'c', 'd'],
                columns=['one', 'two'])
print(df)
print(df.sum)
print(df.sum(axis='columns'))
print(df.mean(axis='columns', skipna=False))
print(df.mean(axis='columns', skipna=True))

print(df.idxmax())
print(df)
print(df.cumsum()) # 积累型方法
print(df.describe())

obj = Series(['a', 'a', 'b', 'c']*4)
print(obj)
print(obj.sort_index())
print(obj.describe())
"""

# 5.3.1相关性和协方差

# 5.3.2唯一值、计数和成员属性
obj = Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
uniques = obj.unique()
print(uniques)
print(obj.value_counts())
print(pd.value_counts(obj.values, sort=False))
mask = obj.isin(['b', 'c'])
print(mask)
print(obj[mask])

to_mach = Series(['c', 'a', 'b', 'b', 'c', 'a'])
unique_vals = Series(['c', 'b', 'a'])
print(pd.Index(unique_vals).get_indexer(to_mach))

data = DataFrame({'Qu1' : [1, 3, 4, 3, 4],
                  'Qu2' : [2, 3, 1, 2, 3],
                  'Qu3' : [1, 5, 2, 4, 4]})
print(data)
print(data.apply(pd.value_counts).fillna(0))
