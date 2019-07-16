import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from datetime import datetime


now = datetime.now()
print(now)
print(now.year, now.month, now.day)

delta = datetime(2011, 1, 7) - datetime(2008, 6, 24, 8, 15)
print(delta)
print(delta.seconds)

from datetime import timedelta
start = datetime(2011, 1, 7)
print(start + timedelta(12))
print(start - 2*timedelta(12))

# 11.1.1字符串与datetime互相转换
stamp = datetime(2011, 1, 3)
print(str(stamp))
print(stamp.strftime('%Y-%m-%d'))

value = '2011-01-03'
print(datetime.strptime(value, '%Y-%m-%d'))
datestrs = ['7/6/2011', '8/6/2011']
# datetime。strtime在已知格式的情况下转换日期的好方式
print([datetime.strptime(x, '%m/%d/%Y') for x in datestrs])

from dateutil.parser import parse

print(parse('2011-01-03'))
print(parse('Jan 31, 1997 10:45 PM'))
print(parse('6/12/2011', dayfirst=True))

datestrs = ['2011-07-06 12:00:00', '2011-08-06 00:00:00']
print(pd.to_datetime(datestrs))

idx = pd.to_datetime(datestrs + [None])
print(idx)
print(idx[2])
print(pd.isnull(idx))

# 11.2时间序列基础
dates = [datetime(2011, 1, 2), datetime(2011, 1, 5), datetime(2011, 1, 7), datetime(2011, 1, 8), datetime(2011, 1, 10), datetime(2011, 1, 12)]
ts = Series(np.random.randn(6), index=dates)
print(ts)
print(ts.index)
print(ts + ts[::2])
print(ts[::2])
print(ts.index.dtype)

stamp = ts.index[0]
print(stamp)

# 11.2.1索引、选择、子集
stamp = ts.index[2]
print(ts[stamp])
print(stamp)
print(ts)
print(ts['2011-01-02'])

longer_ts = Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
print(longer_ts)
print(len(longer_ts))
print(longer_ts['2001'])
print(longer_ts['2001-05'])
print(ts[datetime(2011, 1, 7):])
print(ts['2011-01-06':'2011-01-11'])
print(ts.truncate(after='2011-01-09'))

dates = pd.date_range('1/1/2000', periods=100, freq='W-WED')
long_df = DataFrame(np.random.randn(100, 4), index=dates, columns=['Colorado', 'Texas', 'New York', 'Ohio'])
print(long_df)

# 11.2.2含重复索引的时间序列
dates = pd.DatetimeIndex(['1/1/2000', '1/2/2000', '1/2/2000', '1/2/2000', '1/3/2000'])
dup_ts = Series(np.arange(5), index=dates)
print(dup_ts)
print(dup_ts.index.is_unique)
print(dup_ts['1/3/2000'])   # 不重复
print(dup_ts['1/2/2000'])   # 重复

grouped = dup_ts.groupby(level=0)
print(grouped.mean())
print(grouped.count())


# 11.3日期范围、频率和移位
# 调用resample方法将样本时间序列转换为固定的每日频率数据
print(ts)
resampler = ts.resample('D')
print(resampler)

# 11.3.1生成日期范围
index = pd.date_range('2012-04-01', '2012-06-01')
print(index)
print(len(index))

print(pd.date_range(start='2012-04-01', periods=20))
print(pd.date_range(end='2012-06-01', periods=20))
print(pd.date_range('2000-01-01', '2001-12-01', freq='BM'))
# BM频率 business and of month 月度业务结尾

print(pd.date_range('2012-05-02 12:56:31', periods=5))
print(pd.date_range('2012-05-12 12:56:31', periods=5, normalize=True))


# 11.3.2频率和日期偏置
from pandas.tseries.offsets import Hour, Minute
hour = Hour()
print(hour)
four_hours = Hour(4)
print(four_hours)

print(pd.date_range('2000-01-01', '2000-01-03 23:59', freq='4h'))

# 多个偏置可以通过加法进行联合
print(Hour(2) + Minute(30))

print(pd.date_range('2000-01-01', periods=10, freq='1h30min'))

# 11.3.2.1月中某星期的日期
# week of month 3FRI每月第三个星期五
rng = pd.date_range('2012-01-01', '2012-09-01', freq='WOM-3FRI')
print(list(rng))

#11.3.3移位（前向和后向）日期
ts = Series(np.random.randn(4), index=pd.date_range('1/1/2000', periods=4, freq='M'))
print(ts)
print(ts.shift(2))

# 11.3.3.1使用偏置进行移位日期
from pandas.tseries.offsets import Day, MonthEnd
now = datetime(2011, 11, 17)
print(now + 3 * Day())
print(now + MonthEnd())
print(now + MonthEnd(2))

offset = MonthEnd()
print("This is offset", offset)
# 使用rollforward和rollback分别显示地将日期向前或向后“滚动”
print(offset.rollforward(now))
print(offset.rollback(now))

ts = pd.Series(np.random.randn(20),
               index=pd.date_range('1/15/2000', periods=20, freq='4d'))
print(ts)

print(ts.groupby(offset.rollforward).mean())
print(ts.resample('M').mean())

# 11.4时区处理
import pytz
print(pytz.common_timezones[:])

# 获得pytz的时区对象，可使用pytz。timezone
tz = pytz.timezone('America/New_York')
print("This is tz", tz)

# 11.4.1时区的本地化和转换
rng = pd.date_range('3/9/2012 9:30', periods=6, freq='D')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
print(ts)
print(ts.index.tz)

print(pd.date_range('3/9/2012 9:30', periods=10, freq='D', tz='UTC'))
# 使用tz_localize方法可从简单时区转换到本地化时区
print("This is ts", ts)
ts_utc = ts.tz_localize('UTC')
print("This is us_utc", ts_utc)
print(ts_utc.index)
print(ts_utc.tz_convert('America/New_York'))

# 将其本地化到EST并转换为UTC或柏林时间
ts_eastern = ts.tz_localize('America/New_York')
print(ts_eastern.tz_convert('UTC'))
print(ts_eastern.tz_convert('Europe/Berlin'))
print("Asia shanghai")
print(ts.index.tz_localize('Asia/Shanghai'))
print(ts)

# 11.4.2时区感知时间戳对象的操作
stamp = pd.Timestamp('2011-03-12 04:00')
stamp_utc = stamp.tz_localize('utc')
print(stamp_utc.tz_convert('America/New_York'))

stamp_moscow = pd.Timestamp('2011-03-12 04:00', tz='Europe/Moscow')
print(stamp_moscow)

print(stamp_utc.value)
print(stamp_utc.tz_convert('America/New_York').value)

from pandas.tseries.offsets import Hour
stamp = pd.Timestamp('2012-03-12 01:30', tz='US/Eastern')
print(stamp)
print(stamp  + Hour())

# 构建从DST进行转换前的90分钟
stamp = pd.Timestamp('2012-11-04 00:30', tz='US/Eastern')
print(stamp)
print(stamp + 2 * Hour())

# 11.4.3不同时区的操作
rng = pd.date_range('3/7/2012 9:30', periods=10, freq='B')
ts = Series(np.random.randn(len(rng)), index=rng)
print(ts)

ts1 = ts[:7].tz_localize('Europe/London')
ts2 = ts[2:].tz_localize('Europe/Moscow')

result = ts1 + ts2
print(result)

# 11.5时间区间和区间算术
p = pd.Period(2007, freq='A-DEC')
print(p)
print(p + 5)
print(p - 2)
# 如果两个区间拥有相同的频率，则它们的差是它们之间的单位数
print(pd.Period('2014', freq='A-DEC') - p)
rng = pd.period_range('2000-01-01', '2000-06-30', freq='M')
print(rng)
print(Series(np.random.randn(len(rng)), index=rng))
values = ['2001Q3', '2002Q2', '2003Q1']
index = pd.PeriodIndex(values, freq='Q-DEC')
print(index)

# 11.5.1区间频率转换
p = pd.Period('2007', freq='A-DEC')
print(p)
print(p.asfreq('M', how='start'))
print(p.asfreq('M', how='end'))
p = pd.Period('2007', freq='A-JUN')
print(p)
print(p.asfreq('M', 'start'))
print(p.asfreq('M', 'end'))

p = pd.Period('Aug-2007', 'M')
print(p.asfreq('A-JUN'))
rng = pd.period_range('2006', '2009', freq='A-DEC')
ts = Series(np.random.randn(len(rng)), index=rng)
print(ts)
print(ts.asfreq('M', how='start'))
print(ts.asfreq('B', how='end'))

# 11.5.2季度区间频率
p = pd.Period('2012Q4', freq='Q-JAN')
print(p)
print(p.asfreq('D', 'start'))
print(p.asfreq('D', 'end'))

p4pm = (p.asfreq('B', 'e') - 1).asfreq('T', 's') + 16 * 60
print(p4pm)

rng = pd.period_range('2011Q3', '2012Q4', freq='Q-JAN')
ts = Series(np.arange(len(rng)), index=rng)
print(ts)

new_rng = (rng.asfreq('B', 'e') - 1).asfreq('T', 's') + 16 * 60
ts.index = new_rng.to_timestamp()
print("This is ts", ts)

# 11.5.3将时间戳转换为区间（以及逆转换）
rng = pd.date_range('2000-01-01', periods=3, freq='M')
ts = Series(np.random.randn(len(rng)), index=rng)
print(ts)
pts = ts.to_period()
print(pts)

rng = pd.date_range('1/29/2000', periods=6, freq='D')
ts2 = Series(np.random.randn(6), index=rng)
print(ts2)
print(ts2.to_period('M'))
pts = ts2.to_period()
print(pts)
print(pts.to_timestamp(how='end'))

# 11.5.4从数组生成PeriodIndex
data = pd.read_csv('D:\\test\\pydata-book-2nd-edition\\examples\\macrodata.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print(data.head(5))
print(data.year)
print(data.columns)

index = pd.PeriodIndex(year=data.year, quarter=data.quarter, freq='Q-DEC')
print(index)
data.index = index
print(data.infl)

# 11.6重新采样与频率转换
rng = pd.date_range('2000-01-01', periods=100, freq='D')
ts = Series(np.random.randn(len(rng)), index=rng)
print(ts)
print(ts.resample('M').mean())

# 11.6.1向下采样
rng = pd.date_range('2000-01-01', periods=12, freq='T')
ts = Series(np.random.randn(len(rng)), index=rng)
print(ts)
print("This is resample", ts.resample('5min', closed='right').sum())
print(ts.resample('5min', closed='right', label='right').sum())
print(ts.resample('5min', closed='right', label='right', loffset='-1s').sum())

# 11.6.1开端-峰值-谷值-结束（OHLC）重新采样
print(ts.resample('5min').ohlc())

# 11.6.2向上采样与插值
frame = DataFrame(np.random.randn(2, 4), index=pd.date_range('1/1/2000', periods=2, freq='W-WED'),
                                         columns=['Colorado', 'Texas', 'New York', 'Ohio'])
print(frame)
df_daily = frame.resample('D').asfreq()
print(df_daily)
print(frame.resample('D').ffill())
print(frame.resample('D').ffill(limit=2))
print(frame.resample('W-THU').ffill())

# 11.6.3使用区间进行重新采样
frame = DataFrame(np.random.randn(24, 4),
                  index=pd.period_range('1-2000', '12-2001', freq='M'),
                  columns=['Colorado', 'Texas', 'New York', 'Ohio'])
print(frame)

annual_frame = frame.resample('A-DEC').mean()
print(annual_frame)

# Q-DEC没季度，年末在12月
print(annual_frame.resample('Q-DEC').ffill())
print(annual_frame.resample('Q-DEC', convention='end').ffill())
print(annual_frame.resample('Q-MAR').ffill())

# 11.7移动串口函数
import matplotlib.pyplot as plt

close_px_all = pd.read_csv(open('D:\\test\\pydata-book-2nd-edition\\examples\\stock_px_2.csv'), parse_dates=True, index_col=0)
print(close_px_all.head(5))
print(close_px_all[['AAPL', 'MSFT', 'XOM']].head(5))
close_px = close_px_all[['AAPL', 'MSFT', 'XOM']]
close_px = close_px.resample('B').ffill()
#close_px.AAPL.plot()
#close_px.AAPL.rolling(250).mean().plot()
# 205日滑动窗口分组的而不是直接分组

appl_std250 = close_px.rolling(250, min_periods=10).std()
print(appl_std250.head(5))
#appl_std250.plot()

expanding_mean = appl_std250.expanding().mean()
# 股票价格60日MA
#close_px.rolling(60).mean().plot(logy=True)
print(close_px.rolling('20D').mean().head(10))

"""
# 11.7.1指数加权函数
appl_px = close_px.AAPL['2006':'2007']
ma60 = appl_px.rolling(30, min_periods=20).mean()
ewma60 = appl_px.ewm(span=30).mean()
ma60.plot(style='k--', label='Simple MA')
ewma60.plot(style='k--', label='EW MA')
plt.legend()
"""

# 11.7.2二元移动窗口函数
spx_px = close_px_all['SPX']
spx_rets = spx_px.pct_change()
returns = close_px.pct_change()
corr = returns.AAPL.rolling(125, min_periods=100).corr(spx_rets)

print("corr", corr)
print(corr[0:100])

# 11.7.3用户自定义的移动窗口函数
from scipy.stats import percentileofscore
score_at_2percent = lambda x: percentileofscore(x, 0.02)
result = returns.AAPL.rolling(250).apply(score_at_2percent)
result.plot()
plt.show()
