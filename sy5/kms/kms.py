# #这段代码只能在3.6的python 的环境下运行，因为c=labels.astype(np.float) 已经被弃用了，在3.8及其以后都不采用这样的写法

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.cluster import KMeans
# from sklearn import datasets

# plt.rcParams['font.sans-serif']='simHei'
# plt.rcParams['axes.unicode_minus'] = False


# np.random.seed(5)
# iris = datasets.load_iris()


# x=iris.data
# y=iris.target



# clf = KMeans(n_clusters=3)
# clf.fit(x)
# labels = clf.labels_





# fig = plt.figure(1,figsize=(8,6))
# ax = Axes3D(fig,rect=[0,0,.95,1],elev=48,azim=134)
# ax.scatter(x[:,3],x[:,0],x[:,2],c=labels.astype(np.float),edgecolor='k')
# ax.w_xaxis.set_ticklabels([])
# ax.w_yaxis.set_ticklabels([])
# ax.w_zaxis.set_ticklabels([])
# ax.set_xlabel('花瓣宽度')
# ax.set_ylabel('花瓣长度')
# ax.set_zlabel('花瓣高度')
# ax.set_title('3')
# ax.dist=12
# plt.show()


#适应我的3.11版本，抛弃了弃用的方法
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets

# 设置中文显示
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
np.random.seed(5)
iris = datasets.load_iris()
x = iris.data
y = iris.target

# 使用 KMeans 进行聚类
clf = KMeans(n_clusters=3)
clf.fit(x)
labels = clf.labels_

# 创建 3D 图形
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d', elev=48, azim=134)  # 直接使用 add_subplot
ax.scatter(x[:, 3], x[:, 0], x[:, 2], c=labels.astype(float), edgecolor='k')  # 使用 float 替代 np.float
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('花瓣宽度')
ax.set_ylabel('花瓣长度')
ax.set_zlabel('花瓣高度')
ax.set_title('KMeans Clustering on Iris Dataset')
ax.dist = 12
plt.show()

