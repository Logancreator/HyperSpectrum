import cv2
import os
import spectral
import spectral.io.envi as envi
import matplotlib.pyplot as plt
os.chdir("D:\\UavHyperSpectrum\\Xiongan\\")

retval = os.getcwd()

print("目录修改成功 %s" % retval)

image = envi.open('XiongAn.hdr.txt', 'XiongAn.img')

print(image.shape)

cv2.imshow("windowName", image[:,:,[10,100,200]])

cv2.waitKey(0)
# #图像的显示
# view = imshow(image, (29, 19, 9))
#
# plt.pause(100000)#保持窗口100000秒（有点傻，有别的方法应该）