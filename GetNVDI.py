# -*- coding: utf-8 -*-
"""
This script is to extract the NVDI value from tiff file.

Created on 17/11/2023

@author: JianyeChang
"""

from osgeo import gdal, gdalnumeric, ogr
from PIL import Image, ImageDraw


def imageToArray(i):
    # 将数组转为gdal_array图片
    a = gdalnumeric.numpy.fromstring(i.tobytes(), 'b')
    a.shape = i.im.size[1], i.im.size[0]
    return a


def world2Pixel(geoMatrix, x, y):
    # 计算地理坐标对应的像素坐标
    ulX = geoMatrix[0]
    ulY = geoMatrix[3]
    xDist = geoMatrix[1]
    yDist = geoMatrix[5]
    rtnX = geoMatrix[2]
    rtnY = geoMatrix[4]
    pixel = int((x - ulX) / xDist)
    line = int((ulY - y) / xDist)
    return (pixel, line)


# 影像文件，必须有红光与红外
source = "D:/NDVI-update/farm.tif"
# 输出文件路径
target = "D:/NDVI-update/ndvi.tif"
# 数据转为gdal_array数组
srcArray = gdalnumeric.LoadFile(source)
# 载入图片，获取坐标系
srcImage = gdal.Open(source)
geoTrans = srcImage.GetGeoTransform()
# 定义红光与近红外波段
r = srcArray[1]
ir = srcArray[2]

# 使用shp文件裁剪
field = ogr.Open("D:/NDVI-update/field.shp")
lyr = field.GetLayer("field")
poly = lyr.GetNextFeature()
# 图层坐标转换为像素坐标
minX, maxX, minY, maxY = lyr.GetExtent()
ulX, ulY = world2Pixel(geoTrans, minX, maxY)
lrX, lrY = world2Pixel(geoTrans, maxX, minY)
# 计算输出ndvi图片像素
pxWidth = int(lrX - ulX)
pxHeight = int(lrY - ulY)
# 创建一个空白图片
rClip = r[ulY:lrY, ulX:lrX]
irClip = ir[ulY:lrY, ulX:lrX]
# 为图片创建个geo对象
geoTrans = list(geoTrans)
geoTrans[0] = minX
geoTrans[3] = maxY
# 绘制区域边界
points = []
pixels = []
# 获取多边形几何图形
geom = poly.GetGeometryRef()
pts = geom.GetGeometryRef(0)
# 遍历数组，点转为list对象
for p in range(pts.GetPointCount()):
    points.append((pts.GetX(p), pts.GetY(p)))
# 遍历点集并映射为像素，添加到list中
for p in points:
    pixels.append(world2Pixel(geoTrans, p[0], p[1]))
# 创建栅格图片
rasterPoly = Image.new("L", (pxWidth, pxHeight), 1)
rasterize = ImageDraw.Draw(rasterPoly)
rasterize.polygon(pixels, 0)
# 图片转为gdal_array
mask = imageToArray(rasterPoly)

# 使用遮罩裁剪红光
rClip = gdalnumeric.numpy.choose(mask, \
                                 (rClip, 0)).astype(gdalnumeric.numpy.uint8)
# 使用遮罩裁剪近外红光
irClip = gdalnumeric.numpy.choose(mask, \
                                  (irClip, 0)).astype(gdalnumeric.numpy.uint8)
# 忽略numpy中none值
gdalnumeric.numpy.seterr(all="ignore")
# NDVI = (近外红光 - 红光)/(近外红光 + 红光)
# *1.0 将值转为float
# +1.0 返回出现0做除数的·错误
ndvi = 1.0 * (irClip - rClip) / irClip + rClip + 1.0
# 将所有nan值转为0
ndvi = gdalnumeric.numpy.nan_to_num(ndvi)
# 保存ndvi为geotiff
gdalnumeric.SaveArray(ndvi, target, \
                      format="GTiff", prototype=srcImage)
# 更新图片坐标系和空值
update = gdal.Open(target, 1)
update.SetGeoTransform(list(geoTrans))
update.GetRasterBand(1).SetNoDataValue(0.0)
update = None