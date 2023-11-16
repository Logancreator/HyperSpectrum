'''
文件名: VegetationIndics.py
作者: [Jianye Chang]
日期: 16/11/2023
描述: [这个脚本用于计算各种植被指数]
'''

'''
GMI-1 R750/R550 Gitelson and Merzlyak 1997
GMI-2 R750/R702 Gitelson and Merzlyak 1997
MSR_705 （R750-R450）/（R705-R450） Sims and Gamon 2002
SR_705 R750/R706 Gitelson and Merzlyak 1997
ND_705 （R750-R706）/（ R750+R706） Gitelson and Merzlyak 1997
NDVI-1 （R802-R670）/（ R802+R670） Tucker 1979
NDVI-2 （R682-R554）/（ R682+R554） Gandia et al. 2004
NDVI-3 （R810-R682）/（R810+R682） Rouse et al. 1974
NDVI-4 （R790-R722）/（ R790+R722） Barnes et al. （2000）
PRI （R570-R530）/（ R570+R530） Penuelas et al.1995b
GI R554/R678 Smith et al. 1995
GNDVI （R802-R502）/（ R802+R502） Gitelson et al. 1996
ZM R750/R710 Zarco-Tejada et al.2001
RI-1dB R734/R722 Gupta et al. 2003
RI-2dB R738/R722 Gupta et al. 2003
RI-3dB R742/R718 Gupta et al. 2003
RVI-1 R802/R682 Jordan 1969
RVI-2 R830/R726 秦占飞
VOG-1 R742/R722 Vogelmann et al 1993
VOG-2 （R734-R746）/（R714+R722） Zarco-Tejada et al. （2001）
SIPI （R810-R462）/（ R810+R462） Penuelas et al.（1995）
PPR （R550-R450）/（ R550+R450） Metternich（2003）
NRI （R570-R670）/（ R570+R670） Schleicher et al.（2001）
MCARI-1（R702-R670-0.2（R702-R550））*（R702/R670） Daughtry er al.2000
MCARI-2（R750-R706-0.2（R750-R550））*（R750/R706） Wu et al.2008
MSR-1 （R802-R450）/（R682-R450） Sims and Gamon 2002
MSR-2 （R750/R706）-1/SQRT（（R750/R706）+1） Chen 1996
MTCI-1 （R754-R710）/（R710-R682） Dash and Curran 2004
MTCI-2 （R750-R710）/（R710-R682） Dash 和 Curran （2007）
MSAVI 0.5*（2R802+1-SQRT（（2R802+1）2 -8（R802-R670））） Qi et al. 1994
SAVI （1+0.5）*（R802-R682）（/ R802+R682+0.5） Huete 1988
CI （R678*R690）/（R682*R682） Zarco-Tejada et al.2003
CI_red edge （R842-R870）/（R722-R730）-1 Gutelson et al 2005
CI_green （R842-R870）/R550-1 Gutelson et al 2005
DD （R750-R722）-（R702-R762） le Maire et al. 2004
EVI 2.5（R862-R646）/（1+R862+6R646-7.5R470） Liu and Huete 1995
'''
import cv2
