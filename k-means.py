import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def dis(vet1, vet2): #计算两点之间的欧几里得距离
    return np.sqrt(sum((vet2 - vet1) ** 2))

def img_pre(img): #对图像进行预处理，同时把黑色字块提取出来
    height = img.shape[0]
    long = img.shape[1]
    data = []
    for i in range(height):
        for j in range(long):
            if img[i,j] > 40:
                img[i,j] = 255 #白化处理
            else:
                data.append([j,i]) #x为宽，y为高
    return data,img

def ini_cen(data, k): #初始化各中心点
    long, dim = data.shape
    cen = np.zeros((k, dim)) #存在k个中心点，且列数和原图一致
    for i in range(k): #随机选出k个中心点
         #随机选取一个样本的索引
        index = int(np.random.uniform(0, long)) #随机选取一个样本的索引
        cen[i, :] = data[index, :] #生成初始化的中心点
    return cen

def kmeans(data, k): #K-means本体
    num = data.shape[0] #计算样本个数
    clsData = np.array(np.zeros((num, 2))) #样本的属性，第一列保存该样本属于哪个簇，第二列保存该样本跟它所属簇的误差
    clsChanged = True #决定中心点是否要改变的质量
    cen = ini_cen(data, k) #初始化中心点
    while clsChanged:
        clsChanged = False
        for i in range(num):
            mindis = 100000.0 #初始化最小距离
            minIndex = 0 #定义样本所属的簇
            for j in range(k): #计算每一个中心点与该样本的距离
                distance = dis(cen[j, :], data[i, :]) #如果计算的距离小于最小距离，则更新最小距离
                if distance < mindis:
                    mindis = distance
                    clsData[i, 1] = mindis #更新最小距离
                    minIndex = j #更新样本所属的簇
            if clsData[i, 0] != minIndex: #如果样本的所属的簇发生了变化
                clsChanged = True #中心点要重新计算
                clsData[i, 0] = minIndex #更新样本的簇
        for j in range(k):
            cls_index = np.nonzero(clsData[:, 0] == j) #获取第j个簇所有的样本所在的索引
            pointsIncls = data[cls_index] #第j个簇所有的样本点
            cen[j, :] = np.mean(pointsIncls, axis=0) #计算中心点
    return cen, clsData

def showcls(data, k, cen, clsData): #展示结果
    long, dim = data.shape
    if dim != 2:
        print('输入数据不为二维的！')
        return 1
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'dr', '<r', 'pr'] #用不同颜色形状来表示各个类别
    if k > len(mark):
        print('你选取的k值太大了！')
        return 1
    for i in range(long):
        markIndex = int(clsData[i, 0])
        plt.plot(data[i, 0], data[i, 1], mark[markIndex])
    mark = ['*r', '*b', '*g', '*k', '^b', '+b', 'sb', 'db', '<b', 'pb'] #用不同颜色形状来表示各个类别
    for i in range(k):
        plt.plot(cen[i, 0], cen[i, 1], mark[i], markersize=20)
    plt.show()

img_ori = Image.open('D:\\img.jpg')
img = img_ori.convert('L') #PIL中RGB转灰度图，为I=R*299/1000+G*587/1000+B*114/1000
img.show()
img = np.asarray(img)
data,img = img_pre(img)
plt.imshow(img)
plt.show()
data = np.array(data)
print(data)
k = 4
cen, clsData = kmeans(data, k)
if np.isnan(cen).any():
    print('出错啦！没有黑色区块或者区块不足')
else:
    print('完成！')
showcls(data, k, cen, clsData)
