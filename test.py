import cv2
import os
import numpy as np
import spectral
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from skimage.measure import block_reduce
import matplotlib
import pylab
pylab.show()
matplotlib.use('Agg')

os.chdir("D:\\UavHyperSpectrum\\Xiongan\\")

retval = os.getcwd()

print("目录修改成功 %s" % retval)

image = envi.open('XiongAn.hdr.txt', 'XiongAn.img')

print(image.shape)

#cv2.imshow("windowName", image[:,:,[100,150,50]]/10)

#cv2.waitKey(0)


# 查看两个波段的相关性，间隔10个波段依然具有较高的相似度，可见高光谱数据冗余度很高
def band_density(waveband_1, waveband_2,waveband_1_index,waveband_2_index):
    # 数组太大，先降采样
    waveband_1 = block_reduce(waveband_1, block_size=(10, 10, 1), func=np.mean)
    waveband_2 = block_reduce(waveband_2, block_size=(10, 10, 1), func=np.mean)
    # 计算两个波段的相关性
    correlation = np.corrcoef(waveband_1.flatten(), waveband_2.flatten())[0, 1]
    print("Correlation coefficient between the two bands:", correlation)
    # 生成密度图数据
    xy = np.vstack([waveband_1.flatten(), waveband_2.flatten()])
    density = gaussian_kde(xy)(xy)
    print(density)
    # 绘制散点图
    fig = plt.figure(figsize=(8, 5))
    plt.style.use('seaborn')
    plt.scatter(waveband_1.flatten(), waveband_2.flatten(), c=density, s=1,cmap='jet')
    plt.xlabel('Band ' + str(waveband_1_index))
    plt.ylabel('Band ' + str(waveband_2_index))
    plt.title('Scatter Plot: Band ' + str(waveband_1_index) + ' vs Band ' + str(waveband_2_index))
    # 添加色标
    colorbar = plt.colorbar()
    colorbar.set_label('Density')
    plt.savefig("test.png")

# 使用zoom函数进行1/2重采样
band_A = image[:,:,45]
band_B = image[:,:,55]
band_density(band_A, band_B, 45, 55)