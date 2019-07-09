import numpy as np
import pandas as pd 
from pandas import DataFrame, Series

"""
# 8.1分层索引
data = Series(np.random.randn(9), index=[['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
                                         [1, 2, 3, 1, 3, 1, 2, 2, 3]])
print(data)
print(data.index)
print(data['b'])
print(data['a':'c'])
print("data.loc[:, 2]", data.loc[:, 2])
print("data.unstack()",data.unstack())
print("data.unstck().stck()", data.unstack().stack())

frame = DataFrame(np.arange(12).reshape((4, 3)),
                  index=[['a', 'a', 'b', 'b'],[1, 2, 1, 2]],
                  columns=[['Ohio', 'Ohio', 'Colorado'],
                           ['Green', 'Red', 'Green']])

print(frame)
frame.index.names = ['key1', 'key2']
frame.columns.names = ['state', 'color']
print(frame)
print(frame['Ohio'])

print(frame)

# 8.1.1重排序和层级排序

print("swaplevel", frame.swaplevel('key1', 'key2'))
print(frame.sort_index(level=1))
print(frame.swaplevel(0, 1).sort_index(level=0))

# 8.1.2按层级进行汇总统计
print(frame)
print(frame.sum(level='key2'))
print(frame.sum(level='color', axis=1))
"""

# 8.1.3使用DataFrame的列进行索引
"""
frame = DataFrame({'a' : range(7),
                   'b' : range(7, 0, -1),
                   'c' : ['one', 'one', 'one', 'two', 'two', 'two', 'two'],
                   'd' : [0, 1, 2, 0, 1, 2, 3]})
print(frame)
print(frame.T)
frame2 = frame.set_index(['c', 'd'])
print(frame2)
# 使用c， d作为索引，保留其他的列
print(frame.set_index(['c', 'd'], drop=False))
# reset_index使set_index反操作，分层索引得索引等级会被移动到列中
print(frame2.reset_index())

# 8.2联合与合并数据集
# 8.2.1数据库风格得DataFrame连接
df1 = DataFrame({'key' : ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                 'data1' : range(7)})
df2 = DataFrame({'key' : ['a', 'b', 'd'],
                 'data2' : range(3)})

print(df1)
print(df2)

print(pd.merge(df1, df2, on='key'))

df3 = DataFrame({'lkey' : ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                 'data1': range(7)})
df4 = DataFrame({'rkey' : ['a', 'b', 'd'],
                 'data2': range(3)})
print(df3)
print(df4)
# 默认情况下merge做的是内连接（‘inner’join）,结果中得键使两张表的交集。其他可选的选项有'left','right','outer'
print(pd.merge(df3, df4, left_on='lkey', right_on='rkey'))
# 外连接（outerjoin）是键的并集，联合左连接与右连接的效果
print(pd.merge(df1, df2, how='outer'))
# inner 只对两张表都有的键的交集进行联合
# left  对所有左表的键进行联合
# right 对所有右表的键进行联合
# outer 对两张表都有的键的并集进行联合

df1 = DataFrame({'key' : ['b', 'b', 'a', 'c', 'a', 'b'],
                 'data1': range(6)})

df2 = DataFrame({'key' : ['a', 'b', 'a', 'b', 'd'],
                 'data2' : range(5)})

print(df1.sort_values('key'))
print(df2.sort_values('key'))

print(pd.merge(df1, df2, on='key', how='left'))

left = DataFrame({'key1' : ['foo', 'foo', 'bar'],
                  'key2' : ['one', 'two', 'one'],
                  'lval' : [1, 2, 3]})
right = DataFrame({'key1' : ['foo', 'foo', 'bar', 'bar'],
                   'key2' : ['one', 'one', 'one', 'two'],
                   'rval' : [4, 5, 6, 7]})
print(left)
print(right)
print(pd.merge(left, right, on=['key1', 'key2'], how='outer'))

print(pd.merge(left, right, on='key1'))
print(pd.merge(left, right, on=['key1'], suffixes=('_left', '_right')))
"""

# 8.2.2根据索引合并
"""
left1 = DataFrame({'key' : ['a', 'b', 'a', 'a', 'b', 'c'],
                   'value' : range(6)})
right1 = DataFrame({'group_val' : [3.5, 7]}, index=['a', 'b'])
print(left1)
print(right1)
print(pd.merge(left1, right1, left_on='key', right_index=True))

# 默认的合并方法是连接键相交，可以使用外连接来进行合并
print(pd.merge(left1, right1, left_on='key', right_index=True, how='outer'))

lefth = DataFrame({'key1' : ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
                   'key2' : [2000, 2001, 2002, 2001, 2002],
                   'data' : np.arange(5.)})
righth = DataFrame(np.arange(12).reshape((6, 2)), index=[['Nevada', 'Nevada', 'Ohio', 'Ohio', 'Ohio', 'Ohio'],
                                                         [2001, 2000, 2000, 2000, 2001, 2002]],
                                                  columns=['event1', 'event2'])
print(lefth)
print(righth)

# 以列表的方式指明合并所需多个列表（注意使用how='outer'处理重复的索引值）
print(pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True))

print(pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True, how='outer'))

# 使用两边的索引进行合并
left2 = DataFrame([[1., 2.], [3., 4.], [5., 6.]], index=['a', 'c', 'e'], columns=['Ohio', 'Nevada'])
right2 = DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]], index=['b', 'c', 'd', 'e'], columns=['Missouri', 'Alabama'])
print(left2)
print(right2)
print(pd.merge(left2, right2, how='outer', left_index=True, right_index=True))
print(left2.join(right2, how='outer'))
print(left1.join(right1, on='key'))

another = DataFrame([[7., 8.], [9., 10.], [11., 12.], [16., 17.]], index=['a', 'c', 'e', 'f'], columns=['New York', 'Oregon'])
print(another)
print(left2.join([right2, another]))
print(left2.join([right2, another], how='outer'))
"""

# 8.2.3沿轴向连接
"""
arr = np.arange(12).reshape((3, 4))
print(arr)
print(np.concatenate([arr, arr], axis=1))

s1 = Series([0, 1], index=['a', 'b'])
s2 = Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = Series([5, 6], index=['f', 'g'])
# 用列表中的这些对象调用concat方法会将值和索引黏在一起
print(pd.concat([s1, s2, s3]))
# concat方法一般默认沿着axis=0轴向生效, axis=1是列
print(pd.concat([s1, s2, s3], axis=1))
s4 = pd.concat([s1, s3])
print(s4)
print(pd.concat([s1, s3], axis=1))
# join=inner ’f','g'标签消失了
print("concat", pd.concat([s1, s4], axis=1))
# join_axes指定用于连接轴向的轴
print(pd.concat([s1, s4], join_axes=[['a', 'c', 'b', 'e']], axis=1))

result = pd.concat([s1, s2, s3], keys=['one', 'two', 'three'])
print(result)
print(result.unstack())
print(pd.concat([s1, s2, s3], axis=1, keys=['one', 'two', 'three']))

# 应用于DataFrame
df1 = DataFrame(np.arange(6).reshape((3, 2)), index=['a', 'b', 'c'], columns=['one', 'two'])
df2 = DataFrame(5 + np.arange(4).reshape((2, 2)), index=['a', 'c'], columns=['three', 'four'])
print(df1)
print(df2)
print(pd.concat([df1, df2], axis=1, keys=['level1', 'level2']))
print(pd.concat([df1, df2], axis=1, keys=['level1', 'level2'], names=['upper', 'lower']))

df1 = DataFrame(np.random.randn(3, 4), columns=['a', 'b', 'c', 'd'])
df2 = DataFrame(np.random.randn(2, 3), columns=['b', 'd', 'e'])

print(df1)
print(df2)

print(pd.concat([df1, df2], ignore_index=True))
"""

# 8.2.4联合重叠数据
"""
a = Series([np.nan, 2.5, 0.0, 3.5, 4.5, np.nan], index=['f', 'e', 'd', 'c', 'b', 'a'])
b = Series([0., np.nan, 2., np.nan, np.nan, 5.], index=['a', 'b', 'c', 'd', 'e', 'f'])
print(a.sort_index())
print(b)
print(np.where(pd.isnull(a), b, a))

# combine_first逐列做相同的操作，根据传入的对象来”修补“调用对象的缺失值
print(b.combine_first(a))

df1 = DataFrame({'a' : [1, np.nan, 5, np.nan],
                 'b' : [np.nan, 2., np.nan, 6.],
                 'c' : range(2, 18, 4)})
df2 = DataFrame({'a' : [5., 4., np.nan, 3., 7.],
                 'b' : [np.nan, 3., 4., 6., 8.]})

print(df1)
print(df2)
print(df1.combine_first(df2))
"""

# 8.3重塑和透视
"""
data = DataFrame(np.arange(6).reshape((2, 3)), index=pd.Index(['Ohio', 'Colorado'], name='state'), 
                                               columns=pd.Index(['one', 'two', 'three'], name='number'))
                                   
print(data)

result = data.stack()
print(result)
print(result.unstack())
print(result.unstack(0))
print(result.unstack('state'))

s1 = Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
s2 = Series([4, 5, 6], index=['c', 'd', 'e'])
data2 = pd.concat([s1, s2], keys=['one', 'two'])
# 如果层级中的所有值并未包含于每个子分组中时，拆分可能会引入缺失值
print(data2.unstack())

s1 = Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
s2 = Series([4, 5, 6], index=['c', 'd', 'e'])
data2 = pd.concat([s1, s2], keys=['one', 'two'])
print(data2)
# 默认情况下堆叠会滤出缺失值，因此堆叠拆堆的操作是可逆的
print(data2.unstack())

print(data2.unstack())
print(data2.unstack().stack())
print(data2.unstack().stack(dropna=False))

# 对DataFrame中拆堆中，被拆堆的层级会变为结果中最低的层级
df = DataFrame({'left' : result,
                'right': result+5}, columns=pd.Index(['left', 'right'], name='side'))
print(df)
print(df.unstack('state'))

# 调用stack方法时。我们可以指明需要堆叠的轴向名称
df.unstack('state').stack('side')
"""

# 8.3.2将”长“透视为”宽“
"""
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
data = pd.read_csv(open('D:\\test\\pydata-book-2nd-edition\\examples\\macrodata.csv'))
periods = pd.PeriodIndex(year=data.year, quarter=data.quarter, name='date')
print(data.head(5))
columns = pd.Index(['realgdp', 'infl', 'unemp'], name='item')
data = data.reindex(columns=columns)
print(data.head())
data.index = periods.to_timestamp('D','end')
ldata = data.stack().reset_index().rename(columns={0: 'value'})
print(ldata.head(5))
# 这种数据即所谓的多时间序列的长格式，或称为具有两个或更多个键的其他观测数据
pivoted = ldata.pivot('date', 'item', 'value')
print(pivoted.head(5))

# 传递的前两个值是分别用作行和列索引的列，然后是可选的数值列填充DataFrame.
# 假设你有两个数值列，你想同时进行重塑
ldata['value2'] = np.random.randn(len(ldata))
print(ldata.head(5))
pivoted = ldata.pivot('date', 'item')
print(pivoted.head(5))
print(pivoted['value'][:5])


unstacked = ldata.set_index(['date', 'item']).unstack('item')
print(ldata.columns)
pivoted1 = ldata.pivot('date', 'item', 'value2')
print(pivoted1.columns)
print(pivoted.index)
"""

# 8.3.3将”宽“透视变为”长“
df = DataFrame({'key' : ['foo', 'bar', 'baz'],
                'A'   : [1, 2, 3],
                'B'   : [4, 5, 6],
                'C'   : [7, 8, 9]})
print(df)

melted = pd.melt(df, ['key'])
print(melted)
print(melted.columns)
# pivot方法可以将数据重塑回原先的布局
reshaped = melted.pivot('key', 'variable', 'value')
print(reshaped)

print(reshaped.reset_index())
# 无须任何分组指标
print(pd.melt(df, id_vars=['key'], value_vars=['A', 'B', 'C']))
