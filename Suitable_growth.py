# -*- encoding: utf-8 -*-

import sys
import os
from osgeo import ogr
import osr
import gdal
import matplotlib.pyplot as mpl
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import datetime
import natsort
# %matplotlib inline
import xlrd
import re
from scipy import interpolate
import math


def Raster_Statistic(data_array, data0):
    # 分类别统计
    sum_a = sum(data0[data_array >= 0.7])
    sum_b = sum(data0[(data_array > 0.3) & (data_array < 0.7)])
    sum_c = sum(data0[data_array <= 0.3])
    sum_all = sum_a + sum_b + sum_c
    print ("适宜占比: %.2f" % round((float(sum_a)/float(sum_all)) * 100, 2))
    print ("较适宜占比: %.2f" % round((float(sum_b)/float(sum_all)) * 100, 2))
    print ("不适宜占比: %.2f" % round((float(sum_c)/float(sum_all)) * 100, 2))
    return


def Raster_Statistic_Areas(GeoTransform, xy_number):
    GEO_Radius = 6357#km
    GEO = GeoTransform
    data0 = np.zeros((xy_number[1], xy_number[0]))
    for iLine in range(xy_number[1]):

        y = GEO[3] + GEO[5] * iLine
        pixel_area = np.abs(
            (GEO[1] / 180.0 * np.pi * GEO_Radius * np.cos(y / 180.0 * np.pi)) * \
            (GEO[5] / 180.0 * np.pi * GEO_Radius))

        data0[iLine, :] = pixel_area

    return data0


def interpolation(longs, lats, r_value):

    no_value = -99
    longs = [longs[i] for i, a in enumerate(r_value) if a != no_value]
    lats = [lats[i] for i, a in enumerate(r_value) if a != no_value]
    r_value = [r_value[i] for i, a in enumerate(r_value) if a != no_value]

    longs = np.array(longs)
    lats = np.array(lats)
    r_value = np.array(r_value)

    lats = lats.repeat(4)
    longs = longs.repeat(4)
    r_value = r_value.repeat(4)
    if len(lats) > 16:
        newfunc1 = interpolate.interp2d(longs, lats, r_value, kind='cubic')
    else:
        lats = lats.repeat(4)
        longs = longs.repeat(4)
        r_value = r_value.repeat(4)
        newfunc1 = interpolate.interp2d(longs, lats, r_value, kind='cubic')
    # 103,113,31,40,401,361
    xnew = np.linspace(103, 113, 401)
    ynew = np.linspace(31, 40, 361)
    r_new = newfunc1(xnew, ynew)

    return r_new


def Read_interpolation(file):
    no_value = -9999
    new_data = []
    try:
        all_data = xlrd.open_workbook(file)
        table = all_data.sheets()[0]
        code = table.col_values(0)[1:]
        lon_data = table.col_values(2)[1:]
        lat_data = table.col_values(3)[1:]
        data = table.col_values(4)[1:]

        dd = []
        for i in range(len(data)):
            if (data[i] != no_value):
                dd.append(i)

        code = [code[i] for i in dd]
        lon_data = [float(lon_data[i]) for i in dd]
        lat_data = [float(lat_data[i]) for i in dd]
        data = [float(data[i]) for i in dd]

        new_data = interpolation(lon_data, lat_data, data)
    except:
        print "File %s isn't exist, please check" % file

    # table = all_data.sheets()[0]
    # code = table.col_values(0)[1:]
    # lon_data = table.col_values(2)[1:]
    # lat_data = table.col_values(3)[1:]
    # data = table.col_values(4)[1:]
    #
    # new_data = interpolation(lon_data, lat_data, data)
    return new_data


def calc_Iu(rehum_mean, Iu_param):
    # 计算湿度适宜性值
    data_array = np.zeros((rehum_mean.shape[0], rehum_mean.shape[1]))
    data_array[data_array == 0] = -1

    data_array[(rehum_mean >= Iu_param[2]) | (rehum_mean < Iu_param[3])] = Iu_param[7]
    data_array[(rehum_mean < Iu_param[1]) & (rehum_mean > Iu_param[0])] = Iu_param[5]
    data_array[((rehum_mean >= Iu_param[1]) & (rehum_mean < Iu_param[2])) |
               ((rehum_mean >= Iu_param[3]) & (rehum_mean <= Iu_param[0]))] = Iu_param[6]

    return data_array


def cal_If(wind_mean, If_param):
    #计算风速适宜性值
    data_array = np.zeros((wind_mean.shape[0], wind_mean.shape[1]))
    data_array[data_array == 0] = -1

    data_array[wind_mean <= If_param[0]] = If_param[3]
    data_array[(wind_mean > If_param[0]) & (wind_mean <= If_param[1])] = If_param[4]
    data_array[wind_mean > If_param[1]] = If_param[5]

    return data_array


def cal_It(tem_mean, It_param):
    #计算气温适宜性值
    data_array = np.zeros((tem_mean.shape[0], tem_mean.shape[1]))
    data_array[data_array == 0] = -1

    data_array[(tem_mean >= It_param[2]) | (tem_mean < It_param[3])] = It_param[7]
    data_array[(tem_mean < It_param[1]) & (tem_mean > It_param[0])] = It_param[5]
    data_array[((tem_mean >= It_param[1]) & (tem_mean < It_param[2])) | ((tem_mean >= It_param[3]) & (tem_mean <= It_param[0]))] = It_param[6]

    return data_array


def Crop_Freeze_injury(data_array, crop_name, tem_max, tem_min, tem_plus_1_min, tem_plus_2_min, tem_plus_3_min, Iw_param_2, select_future_time):
    #判断各种作物的低温冻害情况
    '''
    wheat 小麦
    corn 玉米
    pomegranate 石榴
    kiwifruit 猕猴桃
    grape 葡萄
    cherry 樱桃
    '''

    if crop_name.index("pomegranate"):
        data_array[(tem_min < -10) & (tem_plus_1_min < -10) & (tem_plus_2_min < -10)] = Iw_param_2[2]
    elif crop_name.index("kiwifruit"):
        d1 = "12-01"
        e1 = "2-20"
        d2 = "3-26"
        e2 = "4-5"
        d3 = "11-15"
        e3 = "11-20"
        if (select_future_time > ("-").join([select_future_time.split("-")[0], d1])) & (select_future_time < ("-").join([str(int(select_future_time.split("-")[0]) + 1), e1])):
            data_array[tem_min < -15] = Iw_param_2[2]
        elif (select_future_time > ("-").join([select_future_time.split("-")[0], d2])) & (select_future_time < ("-").join([select_future_time.split("-")[0], e2])):
            data_array[tem_min < -1.5] = Iw_param_2[2]
        elif (select_future_time > ("-").join([select_future_time.split("-")[0], d3])) & (select_future_time < ("-").join([select_future_time.split("-")[0], e3])):
            data_array[tem_min < -0.5] = Iw_param_2[2]
    elif crop_name.index("grape"):
        d1 = "12-01"
        e1 = "3-10"
        d2 = "4-1"
        e2 = "4-10"
        if (select_future_time > ("-").join([select_future_time.split("-")[0], d1])) & (select_future_time < ("-").join([str(int(select_future_time.split("-")[0]) + 1), e1])):
            data_array[tem_min < -16] = Iw_param_2[2]
        elif (select_future_time > ("-").join([select_future_time.split("-")[0], d2])) & (select_future_time < ("-").join([select_future_time.split("-")[0], e2])):
            data_array[tem_min < 0] = Iw_param_2[2]
    elif crop_name.index("cherry"):
        data_array[(tem_min < -10) & (tem_plus_1_min < -10) & (tem_plus_2_min < -10)] = Iw_param_2[2]

    return data_array


def calc_Iw(tem_mean, tem_max, tem_min, tem_1_mean, rehum_mean, rehum14, wind_mean, wind_max, wind12, wp_mean, pre_sum, soil_moisture, crop_name, It_param, If_param, Iu_param, Iw_param_1, Iw_param_2, tem_plus_1_min, tem_plus_2_min, tem_plus_3_min, pre_plus_1_sum, pre_plus_2_sum, pre_plus_3_sum, pre_plus_4_sum, wp_plus_1, wp_plus_2, select_future_time):
    #计算天气状况适宜性值
    # 天气现象[2, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 26, 27, 28]
    data_array = np.zeros((tem_max.shape[0], tem_max.shape[1]))
    data_array[data_array == 0] = Iw_param_2[1]

    TK_number = 0
    while TK_number < len(Iw_param_1):
        TK_value = Iw_param_1[TK_number]

        # 阴天1001
        if int(TK_value) == 1001:
            data_array[wp_mean == 2] = Iw_param_2[2]

        # 雷雨1002
        if int(TK_value) == 1002:
            data_array[wp_mean == 4] = Iw_param_2[2]

        # 冰雹1003
        if int(TK_value) == 1003:
            data_array[wp_mean == 5] = Iw_param_2[2]

        # 雨加雪1004
        if int(TK_value) == 1004:
            data_array[wp_mean == 6] = Iw_param_2[2]

        # 暴雨1005
        if int(TK_value) == 1005:
            data_array[wp_mean == 10] = Iw_param_2[2]

        # 雪1006
        if int(TK_value) == 1006:
            data_array[(wp_mean == 13) | (wp_mean == 14) | (wp_mean == 15) | (wp_mean == 16) | (wp_mean == 17) | (wp_mean == 26) | (wp_mean == 27) | (wp_mean == 28)] = Iw_param_2[2]

        # 雪灾1007
        if int(TK_value) == 1007:
            data_array[wp_mean == 17] = Iw_param_2[2]

        # 雨1008
        if int(TK_value) == 1008:
            data_array[(wp_mean == 6) | (wp_mean == 7) | (wp_mean == 8) | (wp_mean == 9) | (wp_mean == 10) | (wp_mean == 11) | (wp_mean == 12)] = Iw_param_2[2]

        # 阴雨1009
        if int(TK_value) == 1009:
            data_array[((wp_mean == 6) | (wp_mean == 7) | (wp_mean == 8) | (wp_mean == 9) | (wp_mean == 10) | (wp_mean == 11) | (wp_mean == 12)) & ((wp_plus_1 == 6) | (wp_plus_1 == 7) | (wp_plus_1 == 8) | (wp_plus_1 == 9) | (wp_plus_1 == 10) | (wp_plus_1 == 11) | (wp_plus_1 == 12)) & ((wp_plus_2 == 6) | (wp_plus_2 == 7) | (wp_plus_2 == 8) | (wp_plus_2 == 9) | (wp_plus_2 == 10) | (wp_plus_2 == 11) | (wp_plus_2 == 12))] = Iw_param_2[2]

        # 暴风雨1010
        if int(TK_value) == 1010:
            data_array[((wp_mean == 10) | (wp_mean == 11) | (wp_mean == 12)) & (wind_max > 17)] = Iw_param_2[2]

        # 高温1011
        if int(TK_value) == 1011:
            data_array[tem_max > 35] = Iw_param_2[2]

        # 倒春寒1012
        if int(TK_value) == 1012:
            data_array[(tem_mean - tem_1_mean) > 2] = Iw_param_2[2]

        # 霜冻1013
        if int(TK_value) == 1013:
            data_array[tem_min < 0] = Iw_param_2[2]

        # 连阴雨1014
        if int(TK_value) == 1014:
            sum_5_pre = pre_sum + pre_plus_1_sum + pre_plus_2_sum + pre_plus_3_sum + pre_plus_4_sum
            # data_array[pre_sum > 0] = Iw_param_2[2]

            pre_sum[pre_sum > 0] = 1
            pre_plus_1_sum[pre_plus_1_sum > 0] = 1
            pre_plus_2_sum[pre_plus_2_sum > 0] = 1
            pre_plus_3_sum[pre_plus_3_sum > 0] = 1
            pre_plus_4_sum[pre_plus_4_sum > 0] = 1
            sum_5_pre1 = pre_sum + pre_plus_1_sum + pre_plus_2_sum + pre_plus_3_sum + pre_plus_4_sum

            data_array[(sum_5_pre > 20) & (sum_5_pre1 >= 4)] = Iw_param_2[2]

        # 大风1015
        if int(TK_value) == 1015:
            data_array[wind_max > 17] = Iw_param_2[2]

        # 干热风1016
        if int(TK_value) == 1016:
            data_array[(tem_max > 35) & (wind12 > 2) & (rehum14 < 30)] = Iw_param_2[2]

        # 低温冻害1017
        if int(TK_value) == 1017:
            data_array = Crop_Freeze_injury(data_array, crop_name, tem_max, tem_min, tem_plus_1_min, tem_plus_2_min, tem_plus_3_min, Iw_param_2, select_future_time)


        # 干旱1018
        if int(TK_value) == 1018:
            data_array[(pre_sum == 0) & (soil_moisture <= 60)] = Iw_param_2[2]

        TK_number = TK_number + 1

    return data_array


def Read_Mean(files):
    #读取所有文件，取均值
    file1 = open(files[0])
    lines = file1.readlines()
    aa = re.split("\s+", lines[2])
    x_number = aa[6]
    y_number = aa[7]
    data_array = np.zeros((int(y_number), int(x_number), len(files)))
    for i in range(0, len(files)):
        file1 = open(files[i])
        lines = file1.readlines()
        l1 = map(lambda x: re.split("\s+", x.strip()), lines[3:])
        l1 = map(lambda x: [float(y) for y in x], l1)
        l2 = np.array(l1, dtype=float)

        data_array[:, :, i] = l2
    data_mean = data_array.mean(axis=2)
    data_array = None
    return data_mean


def Read_Max(files):
    #读取所有文件，取最大值
    file1 = open(files[0])
    lines = file1.readlines()
    aa = re.split("\s+", lines[2])
    x_number = aa[6]
    y_number = aa[7]
    data_array = np.zeros((int(y_number), int(x_number), len(files)))
    for i in range(0, len(files)):
        file1 = open(files[i])
        lines = file1.readlines()
        l1 = map(lambda x: re.split("\s+", x.strip()), lines[3:])
        l1 = map(lambda x: [float(y) for y in x], l1)
        l2 = np.array(l1, dtype=float)

        data_array[:, :, i] = l2
    data_max = data_array.max(axis=2)
    data_array = None
    return data_max


def Read_Min(files):
    #读取所有文件，取最小值
    file1 = open(files[0])
    lines = file1.readlines()
    aa = re.split("\s+", lines[2])
    x_number = aa[6]
    y_number = aa[7]
    data_array = np.zeros((int(y_number), int(x_number), len(files)))
    for i in range(0, len(files)):
        file1 = open(files[i])
        lines = file1.readlines()
        l1 = map(lambda x: re.split("\s+", x.strip()), lines[3:])
        l1 = map(lambda x: [float(y) for y in x], l1)
        l2 = np.array(l1, dtype=float)

        data_array[:, :, i] = l2
    data_max = data_array.min(axis=2)
    data_array = None
    return data_max


def Read_Mean_Max_Min(files):
    #读取所有文件，取均值、最大值、最小值
    file1 = open(files[0])
    lines = file1.readlines()
    aa = re.split("\s+", lines[2])
    x_number = aa[6]
    y_number = aa[7]
    data_array = np.zeros((int(y_number), int(x_number), len(files)))
    for i in range(0, len(files)):
        file1 = open(files[i])
        lines = file1.readlines()
        l1 = map(lambda x: re.split("\s+", x.strip()), lines[3:])
        l1 = map(lambda x: [float(y) for y in x], l1)
        l2 = np.array(l1, dtype=float)

        data_array[:, :, i] = l2
    data_mean = np.nanmean(data_array, axis=2)
    data_min = np.nanmax(data_array, axis=2)
    data_max = np.nanmax(data_array, axis=2)
    data_array = None
    return data_mean, data_min, data_max


def Read_Sum(files):
    #读取所有文件，累和
    file1 = open(files[0])
    lines = file1.readlines()
    aa = re.split("\s+", lines[2])
    x_number = aa[6]
    y_number = aa[7]
    data_array = np.zeros((int(y_number), int(x_number), len(files)))
    for i in range(0, len(files)):
        file1 = open(files[i])
        lines = file1.readlines()
        l1 = map(lambda x: re.split("\s+", x.strip()), lines[3:])
        l1 = map(lambda x: [float(y) for y in x], l1)
        l2 = np.array(l1, dtype=float)

        data_array[:, :, i] = l2
    data_sum = np.nansum(data_array, axis=2)
    data_array = None
    return data_sum


def Read_Wind_Max(files):
    # 读取所有文件，取风速的最大值
    file1 = open(files[0])
    lines = file1.readlines()
    aa = re.split("\s+", lines[2])
    x_number = aa[6]
    y_number = aa[7]
    data_array = np.zeros((int(y_number), int(x_number), len(files)))
    for i in range(0, len(files)):
        file1 = open(files[i])
        lines = file1.readlines()
        l1 = map(lambda x: re.split("\s+", x.strip()), lines[3:])
        l1 = map(lambda x: [float(y) for y in x], l1)
        l2 = np.array(l1, dtype=float)
        u_wind = l2[0:361, :]
        v_wind = l2[361:, :]
        wind_data = np.sqrt(u_wind**2 + v_wind**2)
        data_array[:, :, i] = wind_data
    data_sum = np.nanmax(data_array, axis=2)
    data_array = None
    return data_sum


def Read_Wind_Mean_Max_Min(files):
    # 读取所有文件，取风速的均值
    file1 = open(files[0])
    lines = file1.readlines()
    aa = re.split("\s+", lines[2])
    x_number = aa[6]
    y_number = aa[7]
    data_array = np.zeros((int(y_number), int(x_number), len(files)))
    for i in range(0, len(files)):
        file1 = open(files[i])
        lines = file1.readlines()
        l1 = map(lambda x: re.split("\s+", x.strip()), lines[3:])
        l1 = map(lambda x: [float(y) for y in x], l1)
        l2 = np.array(l1, dtype=float)
        u_wind = l2[0:361, :]
        v_wind = l2[361:, :]
        wind_data = np.sqrt(u_wind**2 + v_wind**2)
        data_array[:, :, i] = wind_data
    data_mean = np.nanmean(data_array, axis=2)
    data_min = np.nanmin(data_array, axis=2)
    data_max = np.nanmax(data_array, axis=2)
    data_array = None
    return data_mean, data_min, data_max


def Read_Pre_Sum(files):
    # 读取所有文件，累和
    file1 = open(files[0])
    lines = file1.readlines()
    aa = re.split("\s+", lines[2])
    x_number = aa[6]
    y_number = aa[7]
    data_array = np.zeros((int(y_number), int(x_number), len(files)))
    for i in range(0, len(files)):
        file1 = open(files[i])
        lines = file1.readlines()
        l1 = map(lambda x: re.split("\s+", x.strip()), lines[4:(int(y_number) + 4)])
        l1 = map(lambda x: [float(y) for y in x], l1)
        l2 = np.array(l1, dtype=float)

        data_array[:, :, i] = l2
    data_sum = np.nansum(data_array, axis=2)
    data_array = None
    return data_sum


def DefinePrj_TmpOut(data_array, GeoTransform, xy_number, tmp_path):
    #定义投影，输出文件tif
    tmp_path_file_aaa = os.path.join(tmp_path, "aaa_test.tif")
    # file_prj = open(Prjfl_path)
    # prj = file_prj.readlines()

    prj = 'GEOGCS["WGS 84",' \
          'DATUM["WGS_1984",' \
          'SPHEROID["WGS 84",6378137,298.257223563,' \
          'AUTHORITY["EPSG","7030"]],' \
          'AUTHORITY["EPSG","6326"]],' \
          'PRIMEM["Greenwich",0,' \
          'AUTHORITY["EPSG","8901"]],' \
          'UNIT["degree",0.01745329251994328,' \
          'AUTHORITY["EPSG","9122"]],' \
          'AUTHORITY["EPSG","4326"]]'
    OutDrv = gdal.GetDriverByName('GTiff')
    OutDat = OutDrv.Create(tmp_path_file_aaa,
                           xy_number[0], xy_number[1], 1,
                           gdal.GDT_Float64, ["COMPRESS=DEFLATE"])
    assert OutDat is not None
    OutDat.SetGeoTransform(GeoTransform)
    OutDat.SetProjection(prj)
    OutDat.GetRasterBand(1).SetNoDataValue(-1)
    OutDat.GetRasterBand(1).WriteArray(data_array)
    OutDat = None


def Mask(input_shp_path, out_path, tmp_path):
    #裁剪
    tmp_path_file_aaa = os.path.join(tmp_path, "aaa_test.tif")

    cutrst = gdal.Warp(out_path, tmp_path_file_aaa, cutlineDSName=input_shp_path, srcNodata=-1, dstNodata=-1,
                       cropToCutline=True, dstSRS="WGS84")
    cutrst.FlushCache()
    cutrst = None
    print ("Mask OK!")
    os.remove(tmp_path_file_aaa)


def Read_Judge(tem_files, tem_1_file, rehum_files, wind_files, wind_12files, wp_files, pre_files, soil_file, crop_name, parameters1, parameters2, tem_files_plus_1, tem_files_plus_2, tem_files_plus_3, pre_files_plus_1, pre_files_plus_2, pre_files_plus_3, pre_files_plus_4, wp_files_plus_1, wp_files_plus_2, rehum_14files, select_future_time):
    # 读取文件数据，均值，判断
    # (105.4918, 0.01, 0.0, 39.58236, 0.0, -0.01)
    # parameters = [0.3, 1, 0, 15, 20, 35, 10, 0.3, 1, 0.5, 0, 5, 9, 0.2, 1, 0.5, 0, 70, 80, 85, 65, 0.2, 1, 0.5, 0]

    '''
    [0] 天况不适宜：1002;04;05;06;10
    [0-2]天气指标比例：0.3, 天气适宜值：1, 天气不适宜值：0
    [3-10] 气温适宜下限：15, 气温适宜上限：20, 气温不适宜上限：35, 气温不适宜下限：10, 气温指标比例：0.3, 气温适宜值：1, 气温较适宜值：0.5, 气温不适宜值：0
    [11-16] 风速适宜上限：5, 风速不适宜下限：9, 风速指标比例：0.2, 风速适宜值：1, 风速较适宜值：0.5, 风速不适宜值：0
    [17-24] 湿度适宜下限：70, 湿度适宜上限：80, 湿度不适宜上限：85, 湿度不适宜下限：65, 湿度指标比例：0.2, 湿度适宜值：1, 湿度较适宜值：0.5, 湿度不适宜值：0
    '''

    file1 = open(tem_files[0])
    lines = file1.readlines()
    aa = re.split("\s+", lines[2])
    GeoTransform = [float(aa[2]), float(aa[0]), 0.0, float(aa[5]), 0.0, -float(aa[1])]
    xy_number = [int(aa[6]), int(aa[7])]

    tem_mean, tem_min, tem_max = Read_Mean_Max_Min(tem_files)
    tem_plus_1_min = Read_Min(tem_files_plus_1)
    tem_plus_2_min = Read_Min(tem_files_plus_2)
    tem_plus_3_min = Read_Min(tem_files_plus_3)
    tem_1_mean = Read_interpolation(tem_1_file)

    soil_moisture = Read_interpolation(soil_file)

    wind_mean, wind_min, wind_max = Read_Wind_Mean_Max_Min(wind_files)
    wind12 = Read_Wind_Max(wind_12files)

    rehum_mean, rehum_min, rehum_max = Read_Mean_Max_Min(rehum_files)
    rehum14 = Read_Mean(rehum_14files)

    wp_mean = Read_Mean(wp_files)

    pre_sum = Read_Pre_Sum(pre_files)
    pre_plus_1_sum = Read_Pre_Sum(pre_files_plus_1)
    pre_plus_2_sum = Read_Pre_Sum(pre_files_plus_2)
    pre_plus_3_sum = Read_Pre_Sum(pre_files_plus_3)
    pre_plus_4_sum = Read_Pre_Sum(pre_files_plus_4)

    wp_plus_1 = Read_Mean(wp_files_plus_1)
    wp_plus_2 = Read_Mean(wp_files_plus_2)


    It_param = [float(i) for i in parameters2[3:11]]
    If_param = [float(i) for i in parameters2[11:17]]
    Iu_param = [float(i) for i in parameters2[17:]]
    Iw_param_1 = [float(i) for i in parameters1.split(";")]
    Iw_param_2 = [float(i) for i in parameters2[0:3]]

    It = cal_It(tem_mean, It_param)
    Iw = calc_Iw(tem_mean, tem_max, tem_min, tem_1_mean, rehum_mean, rehum14, wind_mean, wp_mean, wind_max, wind12, pre_sum, soil_moisture, crop_name, It_param, If_param, Iu_param, Iw_param_1, Iw_param_2, tem_plus_1_min, tem_plus_2_min, tem_plus_3_min, pre_plus_1_sum, pre_plus_2_sum, pre_plus_3_sum, pre_plus_4_sum, wp_plus_1, wp_plus_2, select_future_time)
    If = cal_If(wind_mean, If_param)
    Iu = calc_Iu(rehum_mean, Iu_param)

    It[It == -1] = None
    Iw[Iw == -1] = None
    If[If == -1] = None
    Iu[Iu == -1] = None

    I_all = It * parameters2[7] + Iw * parameters2[0] + If * parameters2[13] + Iu * parameters2[21]

    print ("tem_mean: %.2f" % round(np.nanmean(tem_mean), 2))
    print ("tem_min: %.2f" % round(np.nanmin(tem_min), 2))
    print ("tem_max: %.2f" % round(np.nanmax(tem_max), 2))

    print ("wind_mean: %.2f" % round(np.nanmean(wind_mean), 2))
    print ("wind_min: %.2f" % round(np.nanmin(wind_min), 2))
    print ("wind_max: %.2f" % round(np.nanmax(wind_max), 2))

    print ("rehum_mean: %.2f" % round(np.nanmean(rehum_mean), 2))
    print ("rehum_min: %.2f" % round(np.nanmin(rehum_min), 2))
    print ("rehum_max: %.2f" % round(np.nanmax(rehum_max), 2))

    return I_all, GeoTransform, xy_number


def Check_file(tem_file_list, rehum_file_list, wind_file_list, wp_file_list, prp_file_list, select_time, select_future_time):
    #按照数据Requirement取出相应的数据集
    # 日数据取文件范围：(024\036\048\060\072\084\096\108\120\144\168)
    # 小时数据取文件范围：000~168

    need_wind_number = np.array([24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168])
    need_wp_number = np.array([24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264])

    need_wind_number = np.array(need_wind_number, dtype=int)
    need_wp_number = np.array(need_wp_number, dtype=int)

    select_time = select_time.split("-")
    select_future_time = select_future_time.split("-")
    d1 = datetime.datetime(int(select_time[0]), int(select_time[1]), int(select_time[2]))
    d2 = datetime.datetime(int(select_future_time[0]), int(select_future_time[1]), int(select_future_time[2]))
    days_number = (d2 - d1).days
    hour_start = [(days_number-1)*24, days_number*24]

    need_wind_number_1 = need_wind_number[(need_wind_number > hour_start[0]) & (need_wind_number <= hour_start[1])]
    need_wp_number_1 = need_wp_number[(need_wp_number > hour_start[0]) & (need_wp_number <= hour_start[1])]
    # need_pre_number_1 = need_wp_number[(need_wp_number > hour_start[0]) & (need_wp_number <= hour_start[1])]
    need_tem_number_1 = range(hour_start[0]+1, hour_start[1]+1, 1)


    file_format = "%s%02d%02d20" % (select_time[0][2:4], int(select_time[1]), int(select_time[2]))

    tem_files = []
    rehum_files = []
    wind_files = []
    wp_files = []
    prp_files = []

    tem_aa = [os.path.basename(i) for i in tem_file_list]
    rehum_aa = [os.path.basename(i) for i in rehum_file_list]
    wind_aa = [os.path.basename(i) for i in wind_file_list]
    wp_aa = [os.path.basename(i) for i in wp_file_list]
    prp_aa = [os.path.basename(i) for i in prp_file_list]

    # 提取wp和降水文件
    for i in range(len(need_wp_number_1)):
        file_format_1 = "%s.%03d" % (file_format, need_wp_number_1[i])
        try:
            cc = (wp_aa.index(file_format_1)) & (prp_aa.index(file_format_1))
            wp_files.append(wp_file_list[wp_aa.index(file_format_1)])
            prp_files.append(prp_file_list[prp_aa.index(file_format_1)])
        except:
            print "File prp%s, wp%s isn't exist, please check" % (file_format_1, file_format_1)

    # 提取wind文件
    for i in range(len(need_wind_number_1)):
        file_format_1 = "%s.%03d" % (file_format, need_wind_number_1[i])
        try:
            cc = wind_aa.index(file_format_1)
            wind_files.append(wind_file_list[wind_aa.index(file_format_1)])
        except:
            print "File wind%s isn't exist, please check" % file_format_1

    # 提取tem文件
    for i in range(len(need_tem_number_1)):
        file_format_1 = "%s.%03d" % (file_format, need_tem_number_1[i])

        try:
            cc = tem_aa.index(file_format_1)
            tem_files.append(tem_file_list[tem_aa.index(file_format_1)])
        except:
             continue

    # 提取rehum文件
    for i in range(len(need_tem_number_1)):
        file_format_1 = "%s.%03d" % (file_format, need_tem_number_1[i])
        try:
            cc = rehum_aa.index(file_format_1)
            rehum_files.append(rehum_file_list[rehum_aa.index(file_format_1)])
        except:
            continue


    return tem_files, rehum_files, wind_files, wp_files, prp_files



def Check_Plus_files(tem_file_list, rehum_file_list, prp_file_list, wp_file_list, wind_file_list, select_time, select_future_time):
    #读取温度连续后两天，降水后四天数据

    need_pre_number = np.array([24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264])
    need_wind_number = np.array([24, 36, 60, 84, 108, 132, 156])
    need_rehum_number = np.array([14, 38, 62, 88, 110, 134, 158])


    select_time = select_time.split("-")
    select_future_time = select_future_time.split("-")
    d1 = datetime.datetime(int(select_time[0]), int(select_time[1]), int(select_time[2]))
    d2 = datetime.datetime(int(select_future_time[0]), int(select_future_time[1]), int(select_future_time[2]))
    days_number = (d2 - d1).days
    hour_start = [(days_number-1) * 24, days_number * 24]
    hour_start_plus_1 = [days_number * 24, (days_number+1) * 24]
    hour_start_plus_2 = [(days_number+1) * 24, (days_number+2) * 24]
    hour_start_plus_3 = [(days_number+2) * 24, (days_number+3) * 24]
    hour_start_plus_4 = [(days_number+3) * 24, (days_number+4) * 24]

    need_tem_number_plus_1 = range(hour_start_plus_1[0] + 1, hour_start_plus_1[1] + 1, 1)
    need_tem_number_plus_2 = range(hour_start_plus_2[0] + 1, hour_start_plus_2[1] + 1, 1)
    need_tem_number_plus_3 = range(hour_start_plus_3[0] + 1, hour_start_plus_3[1] + 1, 1)

    need_pre_number_plus_1 = need_pre_number[(need_pre_number > hour_start_plus_1[0]) & (need_pre_number <= hour_start_plus_1[1])]
    need_pre_number_plus_2 = need_pre_number[(need_pre_number > hour_start_plus_2[0]) & (need_pre_number <= hour_start_plus_2[1])]
    need_pre_number_plus_3 = need_pre_number[(need_pre_number > hour_start_plus_3[0]) & (need_pre_number <= hour_start_plus_3[1])]
    need_pre_number_plus_4 = need_pre_number[(need_pre_number > hour_start_plus_4[0]) & (need_pre_number <= hour_start_plus_4[1])]

    need_wp_number_plus_1 = need_pre_number[(need_pre_number > hour_start_plus_1[0]) & (need_pre_number <= hour_start_plus_1[1])]
    need_wp_number_plus_2 = need_pre_number[(need_pre_number > hour_start_plus_2[0]) & (need_pre_number <= hour_start_plus_2[1])]

    need_wind_number12 = need_wind_number[(need_wind_number > hour_start[0]) & (need_wind_number <= hour_start[1])]

    need_rehum_number14 = need_rehum_number[(need_rehum_number > hour_start[0]) & (need_rehum_number <= hour_start[1])]

    file_format = "%s%02d%02d20" % (select_time[0][2:4], int(select_time[1]), int(select_time[2]))
    tem_aa = [os.path.basename(i) for i in tem_file_list]
    prp_aa = [os.path.basename(i) for i in prp_file_list]
    wp_aa = [os.path.basename(i) for i in wp_file_list]
    wind_aa = [os.path.basename(i) for i in wind_file_list]
    rehum_aa = [os.path.basename(i) for i in rehum_file_list]

    tem_files_plus_1 = []
    tem_files_plus_2 = []
    tem_files_plus_3 = []
    pre_files_plus_1 = []
    pre_files_plus_2 = []
    pre_files_plus_3 = []
    pre_files_plus_4 = []
    wp_files_plus_1 = []
    wp_files_plus_2 = []
    wind_12files = []
    rehum_14files = []

    # 提取tem_files_plus_1文件
    for i in range(len(need_tem_number_plus_1)):
        file_format_1 = "%s.%03d" % (file_format, need_tem_number_plus_1[i])

        try:
            cc = tem_aa.index(file_format_1)
            tem_files_plus_1.append(tem_file_list[tem_aa.index(file_format_1)])
        except:
             continue

    # 提取tem_files_plus_2文件
    for i in range(len(need_tem_number_plus_2)):
        file_format_1 = "%s.%03d" % (file_format, need_tem_number_plus_2[i])

        try:
            cc = tem_aa.index(file_format_1)
            tem_files_plus_2.append(tem_file_list[tem_aa.index(file_format_1)])
        except:
             continue

    # 提取tem_files_plus_3文件
    for i in range(len(need_tem_number_plus_3)):
        file_format_1 = "%s.%03d" % (file_format, need_tem_number_plus_3[i])

        try:
            cc = tem_aa.index(file_format_1)
            tem_files_plus_3.append(tem_file_list[tem_aa.index(file_format_1)])
        except:
             continue

    # 降水pre_files_plus_1文件
    for i in range(len(need_pre_number_plus_1)):
        file_format_1 = "%s.%03d" % (file_format, need_pre_number_plus_1[i])
        try:
            cc = prp_aa.index(file_format_1)
            pre_files_plus_1.append(prp_file_list[prp_aa.index(file_format_1)])
        except:
            print "File prp%s isn't exist, please check" % (file_format_1)

    # 降水pre_files_plus_2文件
    for i in range(len(need_pre_number_plus_2)):
        file_format_1 = "%s.%03d" % (file_format, need_pre_number_plus_2[i])
        try:
            cc = prp_aa.index(file_format_1)
            pre_files_plus_2.append(prp_file_list[prp_aa.index(file_format_1)])
        except:
            print "File prp%s isn't exist, please check" % file_format_1

    # 降水pre_files_plus_3文件
    for i in range(len(need_pre_number_plus_3)):
        file_format_1 = "%s.%03d" % (file_format, need_pre_number_plus_3[i])
        try:
            cc = prp_aa.index(file_format_1)
            pre_files_plus_3.append(prp_file_list[prp_aa.index(file_format_1)])
        except:
            print "File prp%s isn't exist, please check" % file_format_1

    # 降水pre_files_plus_4文件
    for i in range(len(need_pre_number_plus_4)):
        file_format_1 = "%s.%03d" % (file_format, need_pre_number_plus_4[i])
        try:
            cc = prp_aa.index(file_format_1)
            pre_files_plus_4.append(prp_file_list[prp_aa.index(file_format_1)])
        except:
            print "File prp%s isn't exist, please check" % file_format_1

    # 天况wp_files_plus_1文件
    for i in range(len(need_wp_number_plus_1)):
        file_format_1 = "%s.%03d" % (file_format, need_wp_number_plus_1[i])
        try:
            cc = wp_aa.index(file_format_1)
            wp_files_plus_1.append(wp_file_list[wp_aa.index(file_format_1)])
        except:
            print "File wp%s isn't exist, please check" % file_format_1

    # 天况wp_files_plus_2文件
    for i in range(len(need_wp_number_plus_2)):
        file_format_1 = "%s.%03d" % (file_format, need_wp_number_plus_2[i])
        try:
            cc = wp_aa.index(file_format_1)
            wp_files_plus_2.append(wp_file_list[wp_aa.index(file_format_1)])
        except:
            print "File wp%s isn't exist, please check" % file_format_1

    # 风速12时wind_12files文件
    for i in range(len(need_wind_number12)):
        file_format_1 = "%s.%03d" % (file_format, need_wind_number12[i])
        file_format_2 = "%s.%03d" % (file_format, need_wind_number12[i]-2)
        try:
            cc = wind_aa.index(file_format_1)
            wind_12files.append(wind_file_list[wind_aa.index(file_format_1)])
        except:
            try:
                cc = wind_aa.index(file_format_2)
                wind_12files.append(wind_file_list[wind_aa.index(file_format_2)])
            except:
                print "File wind%s or wind%s isn't exist, please check" % (file_format_1, file_format_2)

    # 相对湿度14时数据
    for i in range(len(need_rehum_number14)):
        file_format_1 = "%s.%03d" % (file_format, need_rehum_number14[i])
        file_format_2 = "%s.%03d" % (file_format, need_rehum_number14[i]-2)
        try:
            cc = rehum_aa.index(file_format_1)
            rehum_14files.append(rehum_file_list[rehum_aa.index(file_format_1)])
        except:
            try:
                cc = rehum_aa.index(file_format_2)
                rehum_14files.append(rehum_file_list[rehum_aa.index(file_format_2)])
            except:
                print "File rehum%s or rehum%s isn't exist, please check" % (file_format_1, file_format_2)

    return tem_files_plus_1, tem_files_plus_2, tem_files_plus_3, pre_files_plus_1, pre_files_plus_2, pre_files_plus_3, pre_files_plus_4, wp_files_plus_1, wp_files_plus_2, wind_12files, rehum_14files


def dirlist(tem_dir, rehum_dir, wind_dir, wp_dir, prp_dir):
    # 返回文件夹下所有文件
    tem_file_list = []
    rehum_file_list = []
    wind_file_list = []
    wp_file_list = []
    prp_file_list = []
    for filename in os.listdir(tem_dir):
        tem_file_list.append(os.path.join(tem_dir, filename))

    for filename in os.listdir(rehum_dir):
        rehum_file_list.append(os.path.join(rehum_dir, filename))

    for filename in os.listdir(wind_dir):
        wind_file_list.append(os.path.join(wind_dir, filename))

    for filename in os.listdir(wp_dir):
        wp_file_list.append(os.path.join(wp_dir, filename))

    for filename in os.listdir(prp_dir):
        prp_file_list.append(os.path.join(prp_dir, filename))

    return tem_file_list, rehum_file_list, wind_file_list, wp_file_list, prp_file_list


def Check_dir(path, select_time):
    #取出满足时间条件的温度、湿度文件夹路径

    select_time = select_time.split("-")
    tem = "t2m"
    rehum = "rh2m"
    wind = "10uv24"
    wp = "wp24"
    prp = "rain24"

    tem_dir = os.path.join(path, tem, "999", "%4d" % int(select_time[0]), "%2d" % int(select_time[1]), "%2d" % int(select_time[2]))
    rehum_dir = os.path.join(path, rehum, "999", "%4d" % int(select_time[0]), "%2d" % int(select_time[1]), "%2d" % int(select_time[2]))
    wind_dir = os.path.join(path, wind, "999", "%4d" % int(select_time[0]), "%2d" % int(select_time[1]), "%2d" % int(select_time[2]))
    wp_dir = os.path.join(path, wp, "999", "%4d" % int(select_time[0]), "%2d" % int(select_time[1]), "%2d" % int(select_time[2]))
    prp_dir = os.path.join(path, prp, "999", "%4d" % int(select_time[0]), "%2d" % int(select_time[1]), "%2d" % int(select_time[2]))
    return tem_dir, rehum_dir, wind_dir, wp_dir, prp_dir


def main(path, shp_path, out_path, tmp_path, crop_name, select_time, select_future_time, parameters1, parameters2, tem_1_file, soil_file):
    #定位文件夹
    tem_dir, rehum_dir, wind_dir, wp_dir, prp_dir = Check_dir(path, select_time)
    tem_file_list, rehum_file_list, wind_file_list, wp_file_list, prp_file_list = dirlist(tem_dir, rehum_dir, wind_dir, wp_dir, prp_dir)

    # 提取预报当天数据
    tem_files, rehum_files, wind_files, wp_files, pre_files = Check_file(tem_file_list, rehum_file_list, wind_file_list, wp_file_list, prp_file_list, select_time, select_future_time)

    # 提取预报日连续后几日的温度和降水数据
    tem_files_plus_1, tem_files_plus_2, tem_files_plus_3, pre_files_plus_1, pre_files_plus_2, pre_files_plus_3, pre_files_plus_4, wp_files_plus_1, wp_files_plus_2, wind_12files, rehum_14files = Check_Plus_files(tem_file_list, rehum_file_list, prp_file_list, wp_file_list, wind_file_list, select_time, select_future_time)

    # 读取并提取经纬度
    data_array, GeoTransform, xy_number = Read_Judge(tem_files, tem_1_file, rehum_files, wind_files, wind_12files, wp_files, pre_files, soil_file, crop_name, parameters1, parameters2, tem_files_plus_1, tem_files_plus_2, tem_files_plus_3, pre_files_plus_1, pre_files_plus_2, pre_files_plus_3, pre_files_plus_4, wp_files_plus_1, wp_files_plus_2, rehum_14files, select_future_time)

    # 定义投影，输出临时文件
    DefinePrj_TmpOut(data_array, GeoTransform, xy_number, tmp_path)

    # 腌膜、统计
    Mask(shp_path, out_path, tmp_path)
    data_area = Raster_Statistic_Areas(GeoTransform, xy_number)
    Raster_Statistic(data_array, data_area)



if __name__ == '__main__':
    path = "/Users/shuailee/Documents/Climate/Example/dcoef"
    tem_1_file = "/Users/shuailee/Documents/Climate/Example/tem_1_mean.xlsx"
    soil_file = "/Users/shuailee/Documents/Climate/Example/soil_moisture.xlsx"
    shp_path = "/Users/shuailee/Documents/Climate/Example/Boundary/shaanxi.shp"
    tmp_path = "/Users/shuailee/Documents/Climate/tmp"
    out_path = "/Users/shuailee/Documents/Climate/tmp"
    suitable_name = "suitable_12-25.tif"
    crop_name = "wheat"
    select_time = "2017-12-22"
    select_future_time = "2017-12-25"
    parameters1 = "1002;1004;1005;1006;1010"
    parameters2 = [0.3, 1, 0, 15, 20, 35, 10, 0.3, 1, 0.5, 0, 5, 9, 0.2, 1, 0.5, 0, 70, 80, 85, 65, 0.2, 1, 0.5, 0]

    if not os.path.exists(out_path):
        ccc = "mkdir -p %s" % out_path
        os.system(ccc)

    if not os.path.exists(tmp_path):
        ccc = "mkdir -p %s" % tmp_path
        os.system(ccc)

    out_path_file = os.path.join(out_path, suitable_name)

    main(path, shp_path, out_path_file, tmp_path, crop_name, select_time, select_future_time, parameters1, parameters2, tem_1_file, soil_file)
