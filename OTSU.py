# -*- coding: utf-8 -*-
"""
This script is to use OTSU al

Created on 17/11/2023

@author: JianyeChang
"""

import numpy as np
import cv2 as cv

# 将图片转为灰度图
# https://blog.csdn.net/weixin_43635647/article/details/99625105
img = cv.imread('test.jpg', 0)
cv.imshow("img", img)
cv.waitKey()


def OTSU_numpy(img_gray, GrayScale):
    assert img_gray.ndim == 2, "must input a gary_img"  # shape有几个数字, ndim就是多少
    img_gray = np.array(img_gray).ravel().astype(np.uint8)
    u1 = 0.0  # 背景像素的平均灰度值
    u2 = 0.0  # 前景像素的平均灰度值
    th = 0.0

    # 总的像素数目
    PixSum = img_gray.size
    # 各个灰度值的像素数目
    PixCount = np.zeros(GrayScale)
    # 各灰度值所占总像素数的比例
    PixRate = np.zeros(GrayScale)
    # 统计各个灰度值的像素个数
    for i in range(PixSum):
        # 默认灰度图像的像素值范围为GrayScale
        Pixvalue = img_gray[i]
        PixCount[Pixvalue] = PixCount[Pixvalue] + 1

    # 确定各个灰度值对应的像素点的个数在所有的像素点中的比例。
    for j in range(GrayScale):
        PixRate[j] = PixCount[j] * 1.0 / PixSum
    Max_var = 0
    # 确定最大类间方差对应的阈值
    for i in range(1, GrayScale):  # 从1开始是为了避免w1为0.
        u1_tem = 0.0
        u2_tem = 0.0
        # 背景像素的比列
        w1 = np.sum(PixRate[:i])
        # 前景像素的比例
        w2 = 1.0 - w1
        if w1 == 0 or w2 == 0:
            pass
        else:  # 背景像素的平均灰度值
            for m in range(i):
                u1_tem = u1_tem + PixRate[m] * m
            u1 = u1_tem * 1.0 / w1
            # 前景像素的平均灰度值
            for n in range(i, GrayScale):
                u2_tem = u2_tem + PixRate[n] * n
            u2 = u2_tem / w2
            # print(u1)
            # 类间方差公式：G=w1*w2*(u1-u2)**2
            tem_var = w1 * w2 * np.power((u1 - u2), 2)
            # print(tem_var)
            # 判断当前类间方差是否为最大值。
            if Max_var < tem_var:
                Max_var = tem_var  # 深拷贝，Max_var与tem_var占用不同的内存空间。
                th = i
    return th

print("使用numpy的方法：" + str(th))  # 结果为 135
th = OTSU_numpy(img, 256)


def OTSU_cv():
    #该函数返回的第一个值就是输入的thresh值，第二个就是处理后的图像
    img = cv.imread('test.jpg',0)
    retVal, a_img = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
    print("使用opencv函数的方法：" + str(retVal))	# 结果为 134
    cv.imshow("a_img",a_img)
    cv.waitKey()
