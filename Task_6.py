import numpy as np 
import pandas as pd 
from pandas import DataFrame, Series


df = DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
                'key2' : ['one', 'two', 'one', 'two', 'one'],
                'data1': np.random.randn(5),
                'data2': np.random.randn(5)})

print(df)

grouped = df['data1'].groupby(df['key1'])
print(grouped)
# grouped变量现在是一个Groupby对象。
print(grouped.mean())
print(grouped.min())
print(grouped.max())

means = df['data1'].groupby([df['key1'], df['key2']]).mean()
print(means)
means1 = df['data1'].groupby([df['key2'], df['key1']]).mean()
print(means1)
# 包含唯一键对的多层索引
print(means.unstack())

states = np.array(['Ohio', 'California', 'California', 'Ohio', 'Ohio'])
years = np.array([2005, 2005, 2006, 2005, 2006])

print(df)
print(df['data1'].groupby([states, years]).mean())

print(df.groupby('key1').mean())
print(df.groupby(['key1', 'key2']).mean())
print(df.groupby(['key1', 'key2']).size())


# 10.1.1遍历分组
for name, group in df.groupby('key1'):
    print("for")
    print(name)
    print(group)

for (k1, k2), group in df.groupby(['key1', 'key2']):
    print(k1, k2)
    print(group)

pieces = dict(list(df.groupby('key1')))
print(pieces)

print(df.dtypes)

for dtype, group in grouped:
    print(dtype)
    print(group)

# 10.1.2选择一列或所有列的子集
# df.groupby('key1')['data1']
# df.groupby('keu1')['data2']
print(df['data1'].groupby(df['key1']))

print(df.groupby(['key1', 'key2'])[['data2']].mean())
s_grouped = df.groupby(['key1', 'key2'])['data2']
print(s_grouped)
print(s_grouped.mean())

# 10.1.3使用字典和Series分组
people = DataFrame(np.random.randn(5, 5),
                   columns=['a', 'b', 'c','d', 'e'],
                   index=['Joe', 'Steve', 'Wes', 'Jim', 'Travel'])
print(people)
people.iloc[3, [1, 2]] = np.nan 
print(people)
mapping = {'a' : 'red', 'b' : 'red', 'c' : 'blue', 'd' : 'blue', 'e' : 'red', 'f' : 'orange'}
by_column = people.groupby(mapping, axis=1)
print(by_column)
print(by_column.sum())

map_series = Series(mapping)
print(map_series)
print(people.groupby(map_series, axis=1).count())

# 10.1.4使用函数分组
print(people.groupby(len).sum())

key_list = ['one', 'one', 'one', 'two', 'two']
print(people.groupby([len, key_list]).min())

# 10.1.4根据索引层级分组
columns = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'], 
                                     [1, 3, 5, 1, 3]], names=['city', 'tenor'])
print(columns)

hire_df = DataFrame(np.random.randn(4, 5), columns=columns)
print(hire_df)
print(hire_df.groupby(level='city', axis=1).count())

# 10.2数据聚合
print(df)
grouped = df.groupby('key1')
print(grouped['data1'].quantile(0.9))

def peak_to_peak(arr):
    return arr.max() - arr.min()

print(grouped.agg(peak_to_peak))
print(grouped.describe())

# 10.2.1逐列及多函数应用
tips = pd.read_csv(open('D:\\test\\日常\\2019.6.27_Task_3\\tips.csv'))
# 添加总账单的消费比例
print(tips)
tips['tip_pct'] = tips['tip'] / tips['total_bill']
print(tips)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

grouped_pct = tips.groupby(['day', 'smoker'])
print(grouped_pct.agg('mean'))
print(grouped_pct.agg(['mean', 'std', peak_to_peak]))

print(grouped.agg([('foo', 'mean'), ('bar', np.std)]))
functions = ['count', 'mean', 'max']
result = grouped_pct['tip_pct', 'total_bill'].agg(functions)
print(result)
print(result['tip_pct'])

ftuples = [('Durchschnitt', 'mean'), ('Abweichung', np.var)]
print(grouped_pct['tip_pct', 'total_bill'].agg(ftuples))
print(grouped_pct.agg({'tip' : np.max, 'size' : 'sum'}))

# 10.2.2返回不含索引的聚合数据
print(tips.groupby(['day', 'smoker'], as_index=False).mean())
print(tips.groupby(['day', 'smoker'], as_index=True).mean())

# 10.3应用： 通用拆分-应用-联合
def top(df, n=5, column='tip_pct'):
    return df.sort_values(by=column)[-n:]

print(top(tips, n=6))
print(tips.groupby('smoker').apply(top))

print(tips.groupby(['smoker', 'day']).apply(top, n=1, column='total_bill'))
result = tips.groupby('smoker')['tip_pct'].describe()
print(result)
print(result.unstack('smoker'))
f = lambda x: x.describe()
print(grouped.apply(f))

# 10.3.1压缩分组键
print(tips.groupby('smoker', group_keys=False).apply(top))

# 10.3.2分位数与桶分析
frame = DataFrame({'data1' : np.random.randn(1000),
                   'data2' : np.random.randn(1000)})
quartiles = pd.cut(frame.data1, 4)
print(quartiles)

def get_states(group):
    return {'min' : group.min(),
            'max' : group.max(),
            'count': group.count(),
            'mean': group.max(),}

grouped = frame.data2.groupby(quartiles)
print(grouped.apply(get_states).unstack())
# 等长桶： 为了根据分位数样本计算出等大小的桶，则需要使用qcut。通过传递labels=False来获得分位数数值
# 返回分位数数值
grouping = pd.qcut(frame.data1, 10, labels=False)
grouped = frame.data2.groupby(grouping)
print(grouped.apply(get_states).unstack())

# 10.3.3示例： 使用指定分组填充缺失值
s = Series(np.random.randn(6))
s[::2] = np.nan
print(s)
print("s.mean()", s.mean())
s.fillna(s.mean())
print(s.fillna(s.mean()))

states = ['Ohio', 'New York', 'Vermont', 'Florida', 'Oregon', 'Nevada', 'California', 'Idaho']
group_key = ['East'] * 4 + ['West'] * 4
data = Series(np.random.randn(8), index=states)
print(data)

data[['Vermont', 'Nevada', 'Idaho']] = np.nan
print(data)
print(data.groupby(group_key).mean())

fill_mean = lambda g: g.fillna(g.mean())
print(data.groupby(group_key).apply(fill_mean))
fill_values = {'East' : 0.5,
               'West' : -1}
fill_func = lambda g: g.fillna(fill_values[g.name])
print(data.groupby(group_key).apply(fill_func))

# 10。3.4示例： 随机采样与排列
suits = ['H', 'S', 'C', 'D']
card_val = (list(range(1, 11)) + [10] * 3) * 4
base_names = ['A'] + list(range(2, 11)) + ['J', 'K', 'Q']
cards = []
for suit in ['H', 'S', 'C', 'D']:
    cards.extend(str(num) + suit for num in base_names)

deck = Series(card_val, index=cards)
print(deck)
print(deck[:13])

def draw(deck, n=5):
    return deck.sample(n)
print(draw(deck))

# 10.3.5示例： 分组加权平均和相关性
df = DataFrame({'category' : ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
                'data'     : np.random.randn(8),
                'weights'  : np.random.rand(8)})

print(df)
grouped = df.groupby('category')
get_wavg = lambda g: np.average(g['data'], weights=g['weights'])
print(grouped.apply(get_wavg))

close_px = pd.read_csv(open('D:\\test\\日常\\2019.6.27_Task_3\\stock_px_2.csv'), parse_dates=True, index_col=0)
print(close_px.info())
spx_corr = lambda x: x.corrwith(x['SPX'])
# pct_change计算百分比的变化
rets = close_px.pct_change().dropna()
print(rets)
df = DataFrame(np.arange(36).reshape(6, 6))

get_year = lambda x: x.year
by_year  = rets.groupby(get_year)
print(by_year.apply(spx_corr))

# 计算内部相关性
print(by_year.apply(lambda g: g['AAPL'].corr(g['MSFT'])))

# 10.3.6示例： 逐组线性回归
"""
import statsmodels.api as sm
def regress(data, xvars, yvar):
    Y = data[yvar]
    X = data[xvars]
    X['intercept'] = 1.
    result = sm.OLS(Y, X).fit()
    return result.params

print(by_year.apply(regress, 'AAPL', ['SPX']))
"""

# 10.4数据透视表与交叉表
print(tips.pivot_table(index=['day', 'smoker']))
print(tips.pivot_table(['tip_pct', 'size', 'total_bill'], index=['time', 'day'], columns='smoker').stack())
# margins=True来扩充这个表包含部分总计
print(tips.pivot_table(['tip_pct', 'size'], index=['time', 'day'], columns='smoker', margins=True))
print(tips.pivot_table('tip_pct', index=['time', 'smoker'], columns='day', aggfunc=len, margins=True))
print(tips.pivot_table('tip_pct', index=['time', 'size', 'smoker'], columns='day', aggfunc='mean', fill_value=0))

# 10。4.1交叉表：crosstab
print(data)