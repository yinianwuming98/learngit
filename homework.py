# 导入numpy
import numpy as np 
import pandas as pd 

# 打印numpy的版本与配置说明
print(np.__version__)
np.show_config()

# 创建一个长度为10的空向量
x1 = np.zeros(10)
print(x1)

# 如何找到任何一个数组的内存大小
print(x1.size)
print(x1.itemsize)
print("%d bytes" %(x1.size * x1.itemsize))

# 如何从命令行得到numpy中add函数的说明文档
print(np.info(np.add))

# 6.创建一个长度为10并且第五个值为1的空向量
x2 = np.zeros(10)
x2[4] = 5
print(x2)

# 7.创建一个值域范围从10到49的向量
x3 = np.arange(10, 50)
print(x3)
print(len(x3))

# 8。反转一个向量（第一个元素变为最后一个）
x3 = x3[::-1]
print(x3)

# 9.创建一个3*3并且从0到8的矩阵
x4 = np.arange(9).reshape((3, 3))
print(x4)

# 10.找到数组[1,2,0,0,4,0]中非0元素的位置索引 
x5 = np.array([1, 2, 0, 0, 4, 0])
print(x5.nonzero())

# 11. 创建一个 3x3 的单位矩阵
x6 = np.eye(3)
print(x6)

# 12. 创建一个 3x3x3的随机数组 
x7 = np.random.random(27).reshape((3, 3, 3))
x8 = np.random.random((3, 3, 3))
print(x7)
print(x8)

# 13. 创建一个 10x10 的随机数组并找到它的最大值和最小值 
x9 = np.random.random((10, 10))
x9_max, x9_min = x9.max(), x9.min()
print(x9.max())
print(x9.min())
print(x9_max, x9_min)

# 14. 创建一个长度为30的随机向量并找到它的平均值 
x10 = np.random.random(30)
x10_mean = x10.mean()
print(x10_mean)

# 15. 创建一个二维数组，其中边界值为1，其余值为0
x11 = np.zeros(9).reshape((3, 3))
x11[0], x11[-1] = 1, 1
print(x11)

Z = np.ones((10, 10))
Z[1:-1:, 1:-1] = 0
print(Z)

# 16. 对于一个存在在数组，如何添加一个用0填充的边界? 
x16 = np.ones((5, 5))
x16 = np.pad(x16, pad_width=1, mode='constant', constant_values=0)
print(x16)

# 17. 以下表达式运行的结果分别是什么? 
print(0 * np.nan, np.nan == np.nan, np.inf > np.nan, np.nan - np.nan, 0.3 == 3 * 0.1)
print("nan, False, False, nan, False")
print(np.inf)

# 18. 创建一个 5x5的矩阵，并设置值1,2,3,4落在其对角线下方位置
x18 = np.diag(1+np.arange(4), k=-1)
x18 = x18 + np.diag(1+np.arange(4), k=1)
print(x18)

# 19. 创建一个8x8 的矩阵，并且设置成棋盘样式
x19_1 = np.zeros((8, 8), dtype=int)
x19_2 = np.zeros((8, 8), dtype=int)
x19_1[1::2, ::2] = 1
x19_2[::2, 1::2] = 2
print(x19_1, "\n", x19_2)

# 20. 考虑一个 (6,7,8) 形状的数组，其第100个元素的索引(x,y,z)是什么?
x20 = np.zeros((6, 7, 8))
x20[(1, 5, 4)] = 100
print(x20)
print(np.unravel_index(100,(6, 7, 8)))

# 22. 对一个5x5的随机矩阵做归一化
x22 = np.random.random((5, 5))
x22_max, x22_min = x22.max(), x22.min()
x22 = (x22 - x22_min)/(x22_max - x22_min)
print(x22)

# 23. 创建一个将颜色描述为(RGBA)四个无符号字节的自定义dtype？
color = np.dtype([("r", np.ubyte, 1),
                  ("g", np.ubyte, 1),
                  ("b", np.ubyte, 1),
                  ("a", np.ubyte, 1)])
print(color)

# 24. 一个5x3的矩阵与一个3x2的矩阵相乘，实矩阵乘积是什么？
x24_1 = np.arange(15).reshape((5, 3))
x24_2 = np.ones((3, 2))
x24_3 = np.eye(3, 3)
print(np.dot(x24_1, x24_3))
print(x24_3)

# 25. 给定一个一维数组，对其在3到8之间的所有元素取反
x25 = np.arange(16)
x25[(x25 > 3) & (x25 < 8)] *= -1
print(x25)

# 26. 下面脚本运行后的结果是什么? 
"""
print(sum(range(5), -1))
print(range(5))
from numpy import *
print(range(5))
print(sum(range(5), -1))
print("9", "9")
"""

# 27. 考虑一个整数向量Z,下列表达合法的是哪个? 
Z = np.zeros(9).reshape((3, 3))
Z[1, 1] = 2
print(Z)
print("Z**Z" , "\n",Z**Z)
# print("2 << Z >> 2", "\n", 2 << Z >> 2) False
print(-Z)
print("Z <- Z", "\n", Z <- Z)
print("1j*Z", "\n", 1j*Z)
print("Z/1/1", "\n", Z/3/6)
# print("Z<Z>Z", "\n", Z<Z>Z) False

# 28. 下列表达式的结果分别是什么?
print("np.array(0) / np.array(0): ", np.array(0) / np.array(0))
print("np.array(0) // np.array(0): ", np.array(0) // np.array(0))
print("np.array([np.nan]).astype(int).astype(float): ", np.array([np.nan]).astype(int).astype(float))

# 29. 如何从零位对浮点数组做舍入 ?
x29 = np.random.uniform(-10, +10, 10)
print(x29)
print(np.copysign(np.ceil(np.abs(x29)), x29))

# 30. 如何找到两个数组中的共同元素?
x30_1 = np.random.randint(0, 10, 10)
x30_2 = np.random.randint(0, 10, 10)
print(x30_1, "\n", x30_2)
print(np.intersect1d(x30_1, x30_2))

"""
# 31. 如何忽略所有的 numpy 警告(尽管不建议这么做)? 
defaults = np.seterr(all="ignore")
x31 = np.ones(1) / 0
print(x31)
_ = np.seterr(**defaults)
with np.errstate(divide='ignore'):
    x31_1 = np.ones(1) / 0
"""

"""
# 32. 下面的表达式是正确的吗?
print("np.sqrt(-1):", np.sqrt(-1))
print("np.emath.sqrt(-1):", np.emath.sqrt(-1))
print(np.sqrt(-1) == np.emath.sqrt(-1))
"""

# 33. 如何得到昨天，今天，明天的日期? 
yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today     = np.datetime64('today', 'D')
tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
print("Yesterday is " + str(yesterday))
print("Today is " + str(today))
print("Tomorrow is " + str(tomorrow))

# 34. 如何得到所有与2016年7月对应的日期？
print(np.arange('2016-07', '2016-08', dtype='datetime64[D]'))

# 35. 如何直接在位计算(A+B)\*(-A/2)(不建立副本
A = np.ones(3)*1
B = np.ones(3)*2
C = np.ones(3)*3
print(np.add(A, B, out=B))
print(np.divide(A, 2, out=A))
print(np.negative(A, out=A))
print(np.multiply(A, B, out=A))

# 36. 用五种不同的方法去提取一个随机数组的整数部分
x36 = np.random.uniform(0, 10, 10)
print(x36)
print(x36 - x36%1)
print(np.floor(x36))
print(np.ceil(x36)-1)
print(x36.astype(int))
print(np.trunc(x36))

# 37. 创建一个5x5的矩阵，其中每行的数值范围从0到4 
x37 = np.zeros((5, 5))
x37 += np.arange(5)
print(x37)

# 38. 通过考虑一个可生成10个整数的函数，来构建一个数组
def generate():
    for x in range(10):
        yield x
x38 = np.fromiter(generate(), dtype=float, count=-1)
print(x38)

# 39. 创建一个长度为10的随机向量，其值域范围从0到1，但是不包括0和1 
x39 = np.linspace(0, 1, 11, endpoint=False)[1:]
print(x39)

# 40. 创建一个长度为10的随机向量，并将其排序 
x40 = np.random.random(10)
x40.sort()
print("x40", x40)

# 41.对于一个小数组，如何用比 np.sum更快的方式对其求和
x41 = np.arange(10)
print(sum(x41))
print(np.add.reduce(x41))

# 42. 对于两个随机数组A和B，检查它们是否相等
x42_1 = np.random.randint(0, 2, 5)
x42_2 = np.random.randint(0, 2, 5)
equal_1 = np.allclose(x42_1, x42_2)
print(equal_1)
equal_2 = np.array_equal(x42_1, x42_2)
print(equal_2)

"""
# 43. 创建一个只读数组(read-only
x43 = np.zeros(10)
x43.flags.writeable = False
x43[0] = 1
print(x43)
"""

# 44. 将笛卡尔坐标下的一个10x2的矩阵转换为极坐标形式
x44 = np.random.random((10, 2))
x, y = x44[:,0], x44[:,1]
x44_1 = np.sqrt(x**2+y**2)
x44_2 = np.arctan2(y, x)
print(x44_1)
print(x44_2)

# 45. 创建一个长度为10的向量，并将向量中最大值替换为1
x45 = np.random.random(10)
x45[x45.argmax()] = 1
print(x45)

# 46. 创建一个结构化数组，并实现 x 和 y 坐标覆盖 [0,1]x[0,1] 区域 
x46 = np.zeros((5,5), [('x', int), ('y', int)])
x46['x'], x46['y'] = np.meshgrid(np.linspace(0, 10, 5),
                                 np.linspace(0, 10, 5))
print(x46)

# 47. 给定两个数组X和Y，构造Cauchy矩阵C (Cij =1/(xi - yj))
x = np.arange(8)
y = x + 0.5
c = 1.0 / np.subtract.outer(x, y)
print("X:", x)
print("Y", y)
print("C", c)
print(np.linalg.det(c))

# 48.  印每个numpy标量类型的最小值和最大值
for dtype in [np.int8, np.int32, np.int64]:
    print("this is", dtype, "min", np.iinfo(dtype).min)
    print("max", np.iinfo(dtype).max)

for dtype in [np.float32, np.float64]:
    print(np.finfo(dtype).min)
    print(np.finfo(dtype).max)
    print(np.finfo(dtype).eps)

# 50. 给定标量时，如何找到数组中最接近标量的值？
x50_1 = np.arange(100)
x50_2 = np.random.uniform(0, 100)
index = (np.abs(x50_1-x50_2)).argmin()
print(x50_1, "\nx50_1[index]", x50_1[index])

# 51. 创建一个表示位置(x,y)和颜色(r,g,b)的结构化数组
x51 = np.zeros(10, [ ('position', [ ('x', float, 1),
                                    ('y', float, 1)]),
                      ('color',   [ ('r', float, 1),
                                    ('g', float, 1),
                                    ('b', float, 1)])])
print(x51)

# 52. 对一个表示坐标形状为(100,2)的随机向量，找到点与点的距离
x52 = np.random.random((10, 2))
x52_x, x52_y = np.atleast_2d(x52[:, 0], x52[:, 1])
x52_d = np.sqrt( (x52_x-x52_x.T)**2 + (x52_y-x52_y.T)**2)
print("this is 52d", x52_d)

# 53. 如何将32位的浮点数(float)转换为对应的整数(integer)?


# 55. 对于numpy数组，enumerate的等价操作是什么？
x55 = np.arange(9).reshape(3, 3)
for index, value in np.ndenumerate(2):
    print(index, value)
for index in np.ndindex(x55.shape):
    print(index, x55[index])

# 56. 生成一个通用的二维Gaussian-like数组 
x56_x, x56_y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
x56_d = np.sqrt(x56_x*x56_x+x56_y*x56_y)
sigma, mu = 1.0, 0.0
x56_g = np.exp(-( (x56_d-mu)**2 / (2.0 * sigma**2)))
print(x56_g)

# 57. 对一个二维数组，如何在其内部随机放置p个元素? 
n = 10
p = 3
x57 = np.zeros((n, n))
np.put(x57, np.random.choice(range(n*n), p, replace=False), 1)
print(x57)

# 58. 减去一个矩阵中的每一行的平均值 
x = np.random.rand(5, 10)
y = x - x.mean(axis=1).reshape(-1, 1)
print(y)

# 59. 如何通过第n列对一个数组进行排序? 
z = np.random.randint(0, 10, (3, 3))
print(z)
print(z[z[:, 1].argsort()])

# 60. 如何检查一个二维数组是否有空列？
z = np.random.randint(0, 3, (3, 10))
print((~z.any(axis=0)).any())

# 61. 从数组中的给定值中找出最近的值 
x61 = np.random.uniform(0, 1, 10)
x61_1 = 0.5
x61_2 = x61.flat[np.abs(x61 - x61_1).argmin()]
print(x61_2)

# 62. 如何用迭代器(iterator)计算两个分别具有形状(1,3)和(3,1)的数组?
x62_A = np.arange(3).reshape(3, 1)
x62_B = np.arange(3).reshape(1, 3)
x62_it = np.nditer([x62_A, x62_B, None])
for x, y, z in x62_it:
    z[...] = x + y
print(x62_it.operands[2])

# 63. 创建一个具有name属性的数组类
class NameArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', "on name")

x63 = NameArray(np.arange(10), "range_10")
print(x63.name)

# 64. 考虑一个给定的向量，如何对由第二个向量索引的每个元素加1(小心重复的索引)? 
x64_z = np.ones(10)
x64_i = np.random.randint(0, len(x64_z), 20)
x64_z += np.bincount(x64_i, minlength=len(x64_z))
print(x64_z)

# 方法二
"""
np.add.at(x64_z, x64_i, 1)
print(x64_z)
"""

# 65. 根据索引列表(I)，如何将向量(X)的元素累加到数组(F)? 
x65_x = [1, 2, 3, 4, 5, 6]
x65_i = [1, 3, 9, 3, 4, 1]
x65_f = np.bincount(x65_i, x65_x)
print(x65_f)

# 66. 考虑一个(dtype=ubyte) 的 (w,h,3)图像，计算其唯一颜色的数量
w, h = 16, 16
x66_i = np.random.randint(0, 2, (h,w,3)).astype(np.ubyte)
x66_f = x66_i[..., 0]*(256*256) + x66_i[..., 1]*256 + x66_i[..., 2]
n = len(np.unique(x66_f))
print(n)

# 67. 考虑一个四维数组，如何一次性计算出最后两个轴(axis)的和？ 
x67_A = np.random.randint(0, 10, (3, 4, 3, 4))
sum = x67_A.sum(axis=(-2, -1))
print(sum)

# 68. 考虑一个一维向量D，如何使用相同大小的向量S来计算D子集的均值？
x68_D = np.random.uniform(0, 1, 100)
x68_S = np.random.randint(0, 10, 100)
x68_D_sums = np.bincount(x68_S, weights=x68_D)
x68_D_counts = np.bincount(x68_S)
x68_D_means = x68_D_sums / x68_D_counts
print(x68_D_means)

# 69. 如何获得点积 dot prodcut的对角线?
x69_A = np.random.uniform(0, 1, (5, 5))
print(x69_A)
x69_B = np.random.uniform(0, 1, (5, 5))
print(x69_B)
print(np.diag(np.dot(x69_A, x69_B)))

# 70. 考虑一个向量[1,2,3,4,5],如何建立一个新的向量，在这个新向量中每个值之间有3个连续的零？
x70_Z = np.array([1, 2, 3, 4, 5])
x70_nz = 3
x70_Z0 = np.zeros(len(x70_Z) + (len(x70_Z)-1)*(x70_nz))
x70_Z0[::x70_nz+1] + x70_Z
print(x70_Z0)

# 71. 考虑一个维度(5,5,3)的数组，如何将其与一个(5,5)的数组相乘？
x71_A = np.ones((5, 5, 3))
x71_B = np.ones((5, 5))
print(x71_A*x71_B[:, :, None])

# 72. 如何对一个数组中任意两行做交换?
x72_A = np.arange(25).reshape(5, 5)
x72_A[[0, 1]] = x72_A[[1, 0]]
print(x72_A)

# 73. 考虑一个可以描述10个三角形的triplets，找到可以分割全部三角形的line segment
faces = np.random.randint(0, 100, (10, 3))
F = np.roll(faces.repeat(2, axis=1), -1, axis=1)
F = F.reshape(len(F)*3, 2)
F = np.sort(F, axis=1)
G = F.view( dtype=[('p0', F.dtype), ('p1', F.dtype)])
G = np.unique(G)
print(G)

# 74. 给定一个二进制的数组C，如何产生一个数组A满足np.bincount(A)==C(
C = np.bincount([1, 1, 2, 3, 4, 4, 6])
A = np.repeat(np.arange(len(C)), C)
print(A)

# 75. 如何通过滑动窗口计算一个数组的平均数?
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

Z = np.arange(20)

print(moving_average(Z, n=3))

# 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1]) 

from numpy.lib import stride_tricks

def rolling(a, window):
     shape = (a.size - window + 1, window)
     strides = (a.itemsize, a.itemsize)
     return stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = rolling(np.arange(10), 3)

print (Z)


# 77. 如何对布尔值取反，或者原位(in-place)改变浮点数的符号(sign)？
Z = np.random.randint(0, 2, 100)
print(np.logical_not(Z, out=Z))

# 78. 考虑两组点集P0和P1去描述一组线(二维)和一个点p,如何计算点p到每一条线 i (P0[i],P1[i])的距离？
def distance(P0, P1, p):
    T = P1 -P0
    L = (T**2).sum(axis=1)
    U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
    U = U.reshape(len(U), 1)
    D = P0 + U*T - P 
    return np.sqrt(D**2).sum(axis=1)

P0 = np.random.uniform(-10, 10, (10, 2))
P1 = np.random.uniform(-10, 10, (10, 2))
P  = np.random.uniform(-10, 10, ( 1, 2))

print(distance(P0, P1, P))


# 79.考虑两组点集P0和P1去描述一组线(二维)和一组点集P，如何计算每一个点 j(P[j]) 到每一条线 i (P0[i],P1[i])的距离？
P0 = np.random.uniform(-10, 10, (10,2))
P1 = np.random.uniform(-10,10,(10,2))
p = np.random.uniform(-10, 10, (10,2))
print (np.array([distance(P0,P1,p_i) for p_i in p]))


# 80.Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a fill value when necessary)

# Z = np.random.randint(0,10,(10,10))
# shape = (5,5)
# fill  = 0
# position = (1,1)

# R = np.ones(shape, dtype=Z.dtype)*fill
# P  = np.array(list(position)).astype(int)
# Rs = np.array(list(R.shape)).astype(int)
# Zs = np.array(list(Z.shape)).astype(int)

# R_start = np.zeros((len(shape),)).astype(int)
# R_stop  = np.array(list(shape)).astype(int)
# Z_start = (P-Rs//2)
# Z_stop  = (P+Rs//2)+Rs%2

# R_start = (R_start - np.minimum(Z_start,0)).tolist()
# Z_start = (np.maximum(Z_start,0)).tolist()
# R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
# Z_stop = (np.minimum(Z_stop,Zs)).tolist()

# r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
# z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]
# R[r] = Z[z]
# print (Z)
# print (R)



# 81. 考虑一个数组Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14],如何生成一个数组R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ...,[11,12,13,14]]? 
x81 = np.arange(1, 15, dtype=np.uint32)
x81_R = stride_tricks.as_strided(x81, (11, 4), (4, 4))
print(x81_R)


# 82. 计算一个矩阵的秩
x82_Z = np.arange(16).reshape((4, 4))
x82_Z -= 1
print(x82_Z)
U, S, V = np.linalg.svd(x82_Z)
rank = np.sum(S> 1e-10)
print(rank)

# 83. 如何找到一个数组中出现频率最高的值？
x83_Z = np.random.randint(0, 10, 50)
print(np.bincount(x83_Z).argmax())

# 84. 从一个10x10的矩阵中提取出连续的3x3区块
x84_Z = np.random.randint(0, 5, (10, 10))
print(x84_Z)
n = 3
i = 1 + (x84_Z.shape[0]-3)
j = i + (x84_Z.shape[1]-3)
C = stride_tricks.as_strided(x84_Z, shape=(i, j, n, n), strides=x84_Z.strides + x84_Z.strides)
print(C)

# 85. 创建一个满足 Z[i,j] == Z[j,i]的子类 
# class Symetric(np.ndarray):
#     def __setitem__(self, index, value):
#         i,j = index
#         super(Symetric, self).__setitem__((i,j), value)
#         super(Symetric, self).__setitem__((j,i), value)

# def symetric(Z):
#     return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)

# S = symetric(np.random.randint(0,10,(5,5)))
# S[2,3] = 42
# print (S)

# 86. 考虑p个 nxn 矩阵和一组形状为(n,1)的向量，如何直接计算p个矩阵的乘积(n,1)？
p, n = 10, 20
M = np.ones((p, n, n))
V = np.ones((p, n, 1))
S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
print(S)

"""
# 87. 对于一个16x16的数组，如何得到一个区域(block-sum)的和(区域大小为4x4)? 
x87_Z = np.ones((16, 16))
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                        np.arange(0, Z.shape[1], k), axis=1)
print(S)
"""

# 88. 如何利用numpy数组实现Game of Life? 
# def iterate(Z):
#     # Count neighbours
#     N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
#          Z[1:-1,0:-2]                + Z[1:-1,2:] +
#          Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

#     # Apply rules
#     birth = (N==3) & (Z[1:-1,1:-1]==0)
#     survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
#     Z[...] = 0
#     Z[1:-1,1:-1][birth | survive] = 1
#     return Z

# Z = np.random.randint(0,2,(50,50))
# for i in range(100): Z = iterate(Z)
# print (Z)

"""
# 89. 如何找到一个数组的第n个最大值?
x89_Z = np.arange(10000)
np.random.shuffle(x89_Z)
n = set
# slow
print(x89_Z[np.argsort(x89_Z)[-n:]])
# fast
print(x89_Z[np.argpartition(-x89_Z,n)[:n]])
"""

# 90. 给定任意个数向量，创建笛卡尔积(每一个元素的每一种组合
def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape  = (len(x) for x in arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T 

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

print(cartesian(([1, 2, 3], [4, 5], [6, 7])))

# 91. 如何从一个正常数组创建记录数组(record array)?
x91_Z = np.array([("hello", 2.5, 3),
                  ("world", 3.6, 2)])

R = np.core.records.fromarrays(x91_Z.T, 
                               names='col1, col2, col3',
                               formats= 'S8, f8, i8')
print(R)


"""
# 92. 考虑一个大向量Z, 用三种不同的方法计算它的立方
x92_X = np.random.rand()
np.power(x, 3)


print(np.einsum('i, i, i->i', x, x, x))
"""

# 93.考虑两个形状分别为(8,3) 和(2,2)的数组A和B. 如何在数组A中找到满足包含B中元素的行？(不考虑B中每行元素顺序)？ 
x92_A = np.random.randint(0, 5, (8, 3))
x92_B = np.random.randint(0, 5, (2, 2))
x92_C = (x92_A[..., np.newaxis, np.newaxis] == 8)
rows = np.where(x92_C.any((3,1)).all(1))[0]
print(rows)

# 94. 考虑一个10x3的矩阵，分解出有不全相同值的行 (如 [2,2,3]) 
Z = np.random.randint(0,5,(10,3))
print(Z)

E = np.all(Z[:,1:] == Z[:,:-1], axis=1)
U = Z[~E]
print (U)


# 95. 将一个整数向量转换为matrix binary的表现形式 
I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
B = ((I.reshape(-1, 1) & (x**np.arange(8))) != 0).astype(int)
print(B[:, ::-1])

# 96. 给定一个二维数组，如何提取出唯一的(unique)行?
Z = np.random.randint(0, 2, (6, 3))
T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
_, idx = np.unique(T, return_index=True)
uZ = Z[idx]
print(uZ)

# 97.考虑两个向量A和B，写出用einsum等式对应的inner, outer, sum, mul函数
A = np.random.uniform(0, 1, 10)
B = np.random.uniform(0, 1, 10)
print("sum")
print(np.einsum('i->', A))# np.sum(A)

# 98. 考虑一个由两个向量描述的路径(X,Y)，如何用等距样例(equidistant samples)对其进行采样(sample)?
phi = np.arange(0, 10*np.pi, 0.1)
a = 1
x = a*phi*np.cos(phi)
y = a*phi*np.sin(phi)

dr = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
r = np.zeros_like(x)
r[1:] = np.cumsum(dr)                # integrate path
r_int = np.linspace(0, r.max(), 200) # regular spaced path
x_int = np.interp(r_int, r, x)       # integrate path
y_int = np.interp(r_int, r, y)

# 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n.
# X = np.asarray([[1.0, 0.0, 3.0, 8.0],
#                 [2.0, 0.0, 1.0, 1.0],
#                 [1.5, 2.5, 1.0, 0.0]])
# n = 4
# M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
# M &= (X.sum(axis=-1) == n)
# print (X[M])

# 100. 对于一个一维数组X，计算它boostrapped之后的95%置信区间的平均值。
x100 = np.random.randn(100)
x100_N = 1000
idx = np.random.randint(0, x100.size, (x100_N, x100.size))
means = x100[idx].mean(axis=1)
confint = np.percentile(means, [2.5, 97.5])
print(confint)