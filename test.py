from scipy.stats import gaussian_kde
from skimage.measure import block_reduce
import os
import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import matplotlib
import pylab

pylab.show()
matplotlib.use('Agg')

os.chdir("D:\\UavHyperSpectrum\\Xiongan\\")
retval = os.getcwd()
print("目录修改成功 %s" % retval)

image = envi.open('XiongAn.hdr.txt', 'XiongAn.img')
#print(image.shape)
def get_color(number):
    cnames = {
    'aliceblue':            '#F0F8FF','antiquewhite':          '#FAEBD7','aqua':                '#00FFFF',
    'aquamarine':           '#7FFFD4','azure':                '#F0FFFF','beige':                '#F5F5DC',
    'bisque':               '#FFE4C4','black':                '#000000','blanchedalmond':       '#FFEBCD',
    'blue':                 '#0000FF','blueviolet':           '#8A2BE2','brown':                '#A52A2A',
    'burlywood':            '#DEB887','cadetblue':            '#5F9EA0','chartreuse':           '#7FFF00',
    'chocolate':            '#D2691E','coral':                '#FF7F50','cornflowerblue':       '#6495ED',
    'cornsilk':             '#FFF8DC','crimson':              '#DC143C','cyan':                 '#00FFFF',
    'darkblue':             '#00008B','darkcyan':             '#008B8B','darkgoldenrod':        '#B8860B',
    'darkgray':             '#A9A9A9','darkgreen':            '#006400','darkkhaki':            '#BDB76B',
    'darkmagenta':          '#8B008B','darkolivegreen':       '#556B2F','darkorange':           '#FF8C00',
    'darkorchid':           '#9932CC','darkred':              '#8B0000','darksalmon':           '#E9967A',
    'darkseagreen':         '#8FBC8F','darkslateblue':        '#483D8B','darkslategray':        '#2F4F4F',
    'darkturquoise':        '#00CED1','darkviolet':           '#9400D3','deeppink':             '#FF1493',
    'deepskyblue':          '#00BFFF','dimgray':              '#696969','dodgerblue':           '#1E90FF',
    'firebrick':            '#B22222','floralwhite':          '#FFFAF0','forestgreen':          '#228B22',
    'fuchsia':              '#FF00FF','gainsboro':            '#DCDCDC','ghostwhite':           '#F8F8FF',
    'gold':                 '#FFD700','goldenrod':            '#DAA520','gray':                 '#808080',
    'green':                '#008000','greenyellow':          '#ADFF2F','honeydew':             '#F0FFF0',
    'hotpink':              '#FF69B4','indianred':            '#CD5C5C','indigo':               '#4B0082',
    'ivory':                '#FFFFF0','khaki':                '#F0E68C','lavender':             '#E6E6FA',
    'lavenderblush':        '#FFF0F5','lawngreen':            '#7CFC00','lemonchiffon':         '#FFFACD',
    'lightblue':            '#ADD8E6','lightcoral':           '#F08080','lightcyan':            '#E0FFFF',
    'lightgoldenrodyellow': '#FAFAD2','lightgreen':           '#90EE90','lightgray':            '#D3D3D3',
    'lightpink':            '#FFB6C1','lightsalmon':          '#FFA07A','lightseagreen':        '#20B2AA',
    'lightskyblue':         '#87CEFA','lightslategray':       '#778899','lightsteelblue':       '#B0C4DE',
    'lightyellow':          '#FFFFE0','lime':                 '#00FF00','limegreen':            '#32CD32'}
    list=[]
    list.extend(cnames.values())
    return list[:number]
# 查看两个波段的相关性，间隔10个波段依然具有较高的相似度，可见高光谱数据冗余度很高
def band_density(waveband_1, waveband_2,waveband_1_index,waveband_2_index):
    # 数组太大，先降采样，使用block_reduce函数进行卷积重采样
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
    plt.figure(figsize=(8, 5))
    plt.style.use("ggplot")
    plt.scatter(waveband_1.flatten(), waveband_2.flatten(), c=density, s=1,cmap='jet')
    plt.xlabel('Band ' + str(waveband_1_index))
    plt.ylabel('Band ' + str(waveband_2_index))
    plt.title('Scatter Plot: Band ' + str(waveband_1_index) + ' vs Band ' + str(waveband_2_index))
    # 添加色标
    colorbar = plt.colorbar()
    colorbar.set_label('Density')
    plt.savefig("band_density.png", dpi = 600)


band_A = image[:,:,45]
band_B = image[:,:,55]
#band_density(band_A, band_B, 45, 55)

def band_boxplot(band_list):

    all_data = [np.array(image[:,:,x]).flatten() for x in band_list]

    #print(all_data)

    labels = ['band - '+str(i) for i in band_list]  ##柱子横坐标

    fig= plt.figure(figsize=(8, 8))

    # 长方形，默认没有notch
    bplot1 = plt.boxplot(all_data,
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=labels)  # will be used to label x-ticks
    plt.title('Rectangular box plot')
    plt.xlabel(f'Band Number -{len(band_list)}', fontsize=10)
    plt.ylabel(f'Hyperspectral value', fontsize=10)

    ###遍历每个箱子对象
    colors = get_color(len(band_list))  ##定义柱子颜色、和柱子数目一致

    for patch, color in zip(bplot1['boxes'], colors):  ##zip快速取出两个长度相同的数组对应的索引值
        patch.set_facecolor(color)  ##每个箱子设置对应的颜色

    # # 输入波段号，查看不同波段的箱线图
    # band_list = [0,1]
    # data = np.array(image[:,:,band_list])
    # print(data)
    # color = get_color(len(band_list))
    # fig = plt.figure(figsize=(10,4))
    # plt.style.use("ggplot")
    # colors = get_color(len(band_list))
    # plt.boxplot(image[:,:,1].reshape(1,len(band_list)), patch_artist=True, labels=band_list, widths=0.05)
    # plt.title('Box Plot', fontsize= 16)
    # plt.xlabel('Class', fontsize= 14)
    # plt.ylabel(f'Band-{len(band_list)}', fontsize= 14)
    plt.savefig("band_boxplot.png")
band_boxplot([i for i in range(50)])