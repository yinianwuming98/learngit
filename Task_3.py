import numpy as np 
import pandas as pd 
from pandas import DataFrame, Series

# 6.1文本格式数据的读写
"""
df = pd.read_csv(open('D:\\test\\日常\\2019.6.27_Task_3\\ex1.csv'))
print(df)

print(pd.read_table(open('D:\\test\\日常\\2019.6.27_Task_3\\ex1.csv'), sep=','))
print(pd.read_csv(open('D:\\test\\日常\\2019.6.27_Task_3\\ex2.csv'), header=None))
print(pd.read_csv(open('D:\\test\\日常\\2019.6.27_Task_3\\ex2.csv'), names=['a', 'b', 'c', 'd', 'message']))
names = ['a', 'b', 'c', 'd', 'message']
print(pd.read_csv(open('D:\\test\\日常\\2019.6.27_Task_3\\ex2.csv'), names=names, index_col='message'))

parsed = pd.read_csv(open('D:\\test\\日常\\2019.6.27_Task_3\\csv_mindex.csv'), index_col=['key1', 'key2'])
print(parsed)
print(list(open('D:\\test\\日常\\2019.6.27_Task_3\\ex3.txt')))

result = pd.read_table(open('D:\\test\\日常\\2019.6.27_Task_3\\ex3.txt'), sep='\s+')
print(result)
# 运用skiprows跳过第一行、第三行、第四行
print(pd.read_csv(open('D:\\test\\日常\\2019.6.27_Task_3\\ex4.csv'), skiprows=[0, 2, 3]))

result = pd.read_csv(open('D:\\test\\日常\\2019.6.27_Task_3\\ex5.csv'))
print(result)
print(pd.isnull(result))
# na_values选项可以传入一个列表或一组字符串来处理缺失值
result = pd.read_csv(open('D:\\test\\日常\\2019.6.27_Task_3\\ex5.csv'), na_values=['Null'])
print(result)

# 在字典中，每列可以指定不同的缺失值标识
sentinels = {'message': ['foo', 'NA'], 'something': ['two']}
print(pd.read_csv(open('D:\\test\\日常\\2019.6.27_Task_3\\ex5.csv')))
print(pd.read_csv(open('D:\\test\\日常\\2019.6.27_Task_3\\ex5.csv'), na_values=sentinels))
"""

# 6.1.1 分块读入文本文件
"""
# 对pandas的显示设置进行调整
pd.options.display.max_rows = 10
result = pd.read_csv(open('D:\\test\\日常\\2019.6.27_Task_3\\ex6.csv'))
print(result)
# norw读取部分行
print(pd.read_csv(open('D:\\test\\日常\\2019.6.27_Task_3\\ex6.csv'), nrows=5))

# 指定chunksize作为每一块的行数
chunker = pd.read_csv(open('D:\\test\\日常\\2019.6.27_Task_3\\ex6.csv'), chunksize=1000)
print(chunker)
tot = pd.Series([])

# 对key列聚合获得计数值
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(), fill_value=0)
tot = tot.sort_values(ascending=False)
print(tot)
"""

# 6.1.2将数据写入文本格式

"""
data = pd.read_csv(open('D:\\test\\日常\\2019.6.27_Task_3\\ex5.csv'))
print(data)


data.to_csv('D:\\test\\日常\\2019.6.27_Task_3\\out.csv')
import sys
data.to_csv(sys.stout, sep='|')


dates = pd.date_range('1/1/2000', periods=7)
ts = Series(np.arange(7), index=dates)
print(ts)
"""

# 6.1.3使用分割格式
"""
import csv 
f = open('D:\\test\\日常\\2019.6.27_Task_3\\ex7.csv')
reader = csv.reader(f)
print(reader)
for line in reader:
    print(line)
with open('D:\\test\\日常\\2019.6.27_Task_3\\ex7.csv') as f:
    lines = list(csv.reader(f))
header, values = lines[0], lines[1:]
data_dict = {h: v for h, v in zip(header, zip(*values))}
print(data_dict)
data1 = DataFrame(data_dict)
print(data1)

# csv方言选项
class my_dialect(csv.Dialect):
    lineterminator = '\n'
    delimiter = '; '
    quotechar = '"'
    quoting = csv.QUOTE_MINIMAL
reader = csv.reader(f, dialect=my_dialect)
"""

# 6.1.4JSON数据
"""
# 
"""
obj = """
{   "name" : "Wes",
    "places_lived" : ["United States", "Spain", "Germany"],
    "pet" : null,
    "siblings" : [{"name" : "Scott", "age" : 30, "pets" : ["Zeus", "Zuko"]},
                  {"name" : "Katie", "age" : 38, "pets" : ["Sixes", "Stache", "Cisco"]}]
                  }
"""
"""
print(obj)

import json

result = json.loads(obj)
print(result)

arjson = json.dumps(result)
siblings = DataFrame(result['siblings'], columns=['name', 'age'])
print(siblings)

data = pd.read_json(open('D:\\test\\日常\\2019.6.27_Task_3\\example.json'))
print(data)
print(data.to_json())
print(data.to_json(orient='records'))


# 6.1.5XML和HTML：网络抓取
tables = pd.read_html(open('D:\\test\\日常\\2019.6.27_Task_3\\fdic_failed_bank_list.html'))
print(len(tables))
print(tables)
failures = tables[0]
print(failures.head())

close_timestamps = pd.to_datetime(failures['Closing Date'])

print(close_timestamps.dt.year.value_counts())

# 使用lxml.objectify解析XML
"""
"""
from lxml import objectify
path = 'D:\\test\\日常\\2019.6.27_Task_3\\'
parsed = objectify.parse(open())
"""
"""

# 6.2二进制格式
frame = pd.read_csv(open('D:\\test\\日常\\2019.6.27_Task_3\\ex1.csv'))
print(frame)

# pickle读取文件“pickel化”

# 6.2.1使用HDF5格式
"""
"""
frame = DataFrame({'a' : np.random.randn(100)})
store = pd.HDFStore()
"""
"""
#
"""

# 6.2.2读取EXCEL
"""
xlsx = pd.ExcelFile('D:\\test\\日常\\2019.6.27_Task_3\\ex1.xlsx')
print(pd.read_excel(xlsx, 'Sheet1'))

frame = pd.read_excel('D:\\test\\日常\\2019.6.27_Task_3\\ex1.xlsx', 'Sheet1')
print(frame)
writer = pd.ExcelWriter('D:\\test\\日常\\2019.6.27_Task_3\\ex2.xlsx')
frame.to_excel(writer, 'Sheet1')
writer.save()
frame.to_excel('D:\\test\\日常\\2019.6.27_Task_3\\ex2.xlsx')
"""

# 6.3与Web API交互
"""
import requests
url = 'https://api.github.com/repos/pandas-dev/pandas/issues'
resp = requests.get(url)
print(resp)

data = resp.json()
print(data[0]['title'])

issues = DataFrame(data, columns=['number', 'title', 'lables', 'state'])
print(issues)
"""

# 6.4与数据库交互
import sqlite3
query = """
CREATE TABLE test
(a VARCHAR(20), b VARCHAR(20),
 c REAL,        d INTEGER
 );
"""

con = sqlite3.connect('mydata.sqlite')

data = [('Atlanta', 'Georgia', 1.25, 6),
        ('Tallahassee', 'Florida', 2.6, 3),
        ('Sacramento', 'California', 1.7, 5)]
stmt = "INSERT INTO test VALUES(?, ?, ?, ?)"
con.execute(stmt, data)
con.commit()