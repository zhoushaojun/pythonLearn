import numpy as np

# (一)数组模块
# Array 基础操作
c = [[1, 2], [3, 4]]  # 二维列表
d = np.array(c)  # 二维numpy数组
d.shape  # (2, 2)
d.size  # 4
print(d.max(axis=0))  # 找维度0，也就是最后一个维度上的最大值，array([3, 4]))
print(d.max(axis=1))  # 找维度1，也就是倒数第二个维度上的最大值，array([2, 4])
print(d.mean(axis=0))  # 找维度0，也就是第一个维度上的均值，array([ 2.,  3.])
d.flatten()  # 展开一个numpy数组为1维数组，array([1, 2, 3, 4])
np.ravel(c)  # 展开一个可以解析的结构为1维数组，array([1, 2, 3, 4])

# Array 产生方式
a = np.array([3, 4])
print(a)
b = np.ones((3, 3), dtype=np.float)
print(b)
c = np.repeat(3, 4)
print(c)
d = np.zeros((2, 2, 3), dtype=np.uint8)
print(d)

e = np.arange(10)
print(e)
f = np.linspace(0, 6, 5)
print(f)

# Array变换
a = np.arange(24).reshape(2, 3, 4)
print(a)

b = a[:, 2, :]
print(b)

c = a[:, :, 1]
print(c)

d = a[:, 1:, 1:-1]
print(d)

h = np.split(np.arange(9), [2, -3])
print(h)

#切片
print("slice")
arr = np.arange(12).reshape((3, 4))
print(arr)
print(arr[0][1])
print(arr[0,:])
print(arr[:,0:2])
print(arr[0:1,:])
print(arr[:,:1])

z = np.arange(6).reshape(2,3)
print(z)
print(z[:,np.newaxis,:])
print(z[:,np.newaxis,:].shape)
print(z[:,np.newaxis,1])

print("-c -r")
a=np.array([1,2])
b=np.array([3,4])
print(np.c_[a,b])
print(np.r_[a,b])

#拼接
print("concrete")
l0 = np.arange(6).reshape((2, 3))
l1 = np.arange(6, 12).reshape((2, 3))
m = np.vstack((l0, l1))
p = np.hstack((l0, l1))
q = np.concatenate((l0, l1))
r = np.concatenate((l0, l1), axis=-1)
#增加维度
s = np.stack((l0, l1))
#转置
t = s.transpose((2, 0, 1))
u = a[0].transpose()

#旋转
u=np.array([[ 0,  4,  8],
       [ 1,  5,  9],
       [ 2,  6, 10],
       [ 3,  7, 11]])
y = np.roll(u, 1)
z = np.roll(u, 1, axis=1)

print(y)
print(z)
# (二)线性模块
a = np.array([3, 4])
b = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print("here")
print(b[0][:4])
c = np.array([1, 0, 1])
print(np.dot(c, b))
print(np.dot(b, c))
print(np.trace(b))

print(np.linalg.norm(a))

print(np.linalg.det(b))
print(np.linalg.matrix_rank(b))
print(np.trace(b))

e = np.array([
    [1, 2],
    [3, 4]
])

# linalg

# 对不镇定矩阵，进行SVD分解并重建
U, s, V = np.linalg.svd(e)

S = np.array([
    [s[0], 0],
    [0, s[1]]
])

print(U, s, V)
print(S)

# (三)随机模块
# 随机模块
import numpy.random as random
print("---Random---")
# 设置随机数种子
random.seed(42)
print(random.rand(3, 3))

print(random.uniform(1,6,10))
print(random.randint(1,6,10))

# 产生2x5的标准正态分布样本
print("normal")
print(random.normal(size=(5,2)))
#二项分布
print(random.binomial(n=5,p=0.5,size=5))

a= np.arange(10)
print(random.choice(a,7))
print(random.choice(a,7,replace=False))
print(random.permutation(a))
print(random.shuffle(a))


# (四) matplotlib
'''
import matplotlib as mpl
import matplotlib.pyplot as plt

# 通过rcParams设置全局横纵轴字体大小
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

np.random.seed(42)
x = np.linspace(0, 5, 100)

y = 2 * np.sin(x) + 0.3 * x ** 2
y_data = y + np.random.normal(scale=0.3, size=100)

plt.figure("data")
plt.plot(x, y_data, '.')

plt.figure("model")
plt.plot(x, y)

plt.figure("data&model")
plt.plot(x, y, 'k', lw=3)
plt.scatter(x, y_data)

plt.savefig('result.png')

#plt.show()

n_samples = 500
dim = 3
samples = np.random.multivariate_normal(
    np.zeros(dim),
    np.eye(dim),
    n_samples
)
print(samples)
print(np.eye(3))

from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)

# 采样个数500
n_samples = 500
dim = 3

# 先生成一组3维正态分布数据，数据方向完全随机
samples = np.random.multivariate_normal(
    np.zeros(dim),
    np.eye(dim),
    n_samples
)

# 通过把每个样本到原点距离和均匀分布吻合得到球体内均匀分布的样本
for i in range(samples.shape[0]):
    r = np.power(np.random.random(), 1.0/3.0)
    samples[i] *= r / np.linalg.norm(samples[i])

upper_samples = []
lower_samples = []

for x, y, z in samples:
    # 3x+2y-z=1作为判别平面
    if z > 3*x + 2*y - 1:
        upper_samples.append((x, y, z))
    else:
        lower_samples.append((x, y, z))

fig = plt.figure('3D scatter plot')
ax = fig.add_subplot(111, projection='3d')

uppers = np.array(upper_samples)
lowers = np.array(lower_samples)

# 用不同颜色不同形状的图标表示平面上下的样本
# 判别平面上半部分为红色圆点，下半部分为绿色三角
ax.scatter(uppers[:, 0], uppers[:, 1], uppers[:, 2], c='r', marker='o')
ax.scatter(lowers[:, 0], lowers[:, 1], lowers[:, 2], c='g', marker='^')

plt.show()
'''

c1 = np.array([1, 2, 3]).reshape((1,3,1))
c2 = np.array([-1, -2, -3, -4, -5]).reshape((1,1,5))
print((c1,c2))


mm=np.array([1,1,1])
pp=np.array([2,2,2])
print(mm*2)
print(mm**2)
print(np.dot(mm,pp))

mm = np.mat([1,2,3])
pp = np.mat([4,5,6])
print(mm*pp.T)
print(np.multiply(mm,pp))

print("--")
transArray= np.arange(24).reshape((3,2,4))
print(transArray)
print("--")
transArray=np.transpose(transArray,(1,0,2))
print(transArray)
print(transArray[-1])

transArray = np.arange(0,1000).reshape(50,20)
print(transArray[:,:,np.newaxis][0])