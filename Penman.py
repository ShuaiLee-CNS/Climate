# -*- encoding: utf-8 -*-

from __future__ import division
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
from xlrd import xldate_as_tuple
import math

# from mpl_toolkits.basemap import Basemap
# from mpl_toolkits.basemap import interp
from scipy import interpolate
import pandas as pd
# import pyKriging
# from pyKriging.krige import kriging
# from pyKriging.samplingplan import samplingplan


def Write2CSV(tmp_path, longs, lats, r_value):
    # 把数据写入到CSV中，以逗号分隔
    tmp_CSV_file_name = os.path.join(tmp_path, "test_inter.csv")
    dataframe = pd.DataFrame({'longs': longs, 'lats': lats, 'r_value': r_value})
    dataframe.to_csv(tmp_CSV_file_name, columns=('longs', 'lats', 'r_value'), index=False, sep=',')


def save(filename, contents):
    # vrt写入
    fh = open(filename, 'w')
    fh.write(contents)
    fh.close()


def Reed_CSV(tmp_path):
    tmp_CSV_file_name = os.path.join(tmp_path, "test_inter.csv")
    ogr.UseExceptions()
    X_name = "longs"
    Y_name = "lats"
    Z_name = "r_value"
    contents = "<OGRVRTDataSource> <OGRVRTLayer name=\"test_inter\"> <SrcDataSource>%s</SrcDataSource> <SrcLayer>test_inter</SrcLayer> <GeometryType>wkbPoint</GeometryType> <LayerSRS>WGS84</LayerSRS> <GeometryField encoding=\"PointFromColumns\" x=\"%s\" y=\"%s\" z=\"%s\" /> </OGRVRTLayer> </OGRVRTDataSource>" % (tmp_CSV_file_name, X_name, Y_name, Z_name)
    filename = os.path.join(tmp_path, "test_inter.vrt")
    save(filename, contents)


def interpolation(shp_path, out_path, tmp_path, longs, lats, r_value):
    Write2CSV(tmp_path, longs, lats, r_value)
    Reed_CSV(tmp_path)
    xnew = [105.487122, 111.241882]
    ynew = [31.706726, 39.585327]

    filename = os.path.join(tmp_path, "test_inter.vrt")
    tmp_path_file_aaa = os.path.join(tmp_path, "aaa_test.tif")
    cutrst = gdal.Grid(destName=tmp_path_file_aaa, srcDS=filename, format='GTiff', width=300, height=500,
                       outputBounds=[xnew[0], ynew[1], xnew[1], ynew[0]], outputSRS="WGS84", noData=-100,
                       algorithm="invdist")
    cutrst.FlushCache()
    cutrst = None
    print("Invdist OK!")
    gdal.Warp(out_path, tmp_path_file_aaa, cutlineDSName=shp_path, dstNodata=-99,
              cropToCutline=True, dstSRS="WGS84")
    # os.remove(tmp_path_file_aaa)
    # os.remove(filename)

    print("Mask OK!")


def interpolation1(longs, lats, r_value):
    '''
    map = Basemap(projection='merc', llcrnrlat=33.69610595700004, urcrnrlat=34.74407959000006,
                   llcrnrlon=107.65850830100004, urcrnrlon=109.82287597700008, lat_ts=20, resolution='c')

    lons = np.array(lon_lat[:, 0])
    lats = np.array(lon_lat[:, 1])
    lons_1 = natsort.natsorted(np.array(lon_lat[:, 0]))
    lats_1 = natsort.natsorted(np.array(lon_lat[:, 1]))
    data = np.array(r_value)

    # x, y = map(lons, lats)

    x2 = np.linspace(lons_1[0][0], lons_1[0][-1], 100)
    y2 = np.linspace(lats_1[0][0], lats_1[-1][0], 100)


    x2, y2 = np.meshgrid(x2, y2)

    grid = interp(data, lons_1[0], np.flipud(lats_1[:, 0]), x2, np.flipud(y2), order=3)
    '''

    '''
    # The Kriging model starts by defining a sampling plan, we use an optimal Latin Hypercube here
    sp = samplingplan(2)
    X = sp.optimallhc(20)

    # Next, we define the problem we would like to solve
    testfun = pyKriging.testfunctions().branin
    y = testfun(X)

    # Now that we have our initial data, we can create an instance of a Kriging model
    k = kriging(X, y, testfunction=testfun, name='simple')
    k.train()

    # Now, five infill points are added. Note that the model is re-trained after each point is added
    numiter = 5
    for i in range(numiter):
        print 'Infill iteration {0} of {1}....'.format(i + 1, numiter)
        newpoints = k.infill(1)
        for point in newpoints:
            k.addPoint(point, testfun(point)[0])
        k.train()

    # And plot the results
    k.plot()
    '''
    # 计算100*100的网格上的插值
    # xnew = np.linspace(lons_1[0][0], lons_1[0][-1], 100)  # x
    # ynew = np.linspace(lats_1[0][0], lats_1[-1][0], 100)  # y
    # fnew = func2(xnew, ynew)
    # gdal.Grid()

    no_value = -9999
    longs = [longs[i] for i, a in enumerate(r_value) if a != no_value]
    lats = [lats[i] for i, a in enumerate(r_value) if a != no_value]
    r_value = [r_value[i] for i, a in enumerate(r_value) if a != no_value]

    longs = np.array(longs)
    lats = np.array(lats)
    r_value = np.array(r_value)

    lats = lats.repeat(4)
    longs = longs.repeat(4)
    r_value = r_value.repeat(4)

    xnew = np.linspace(107.65850830100004, 109.82287597700008, 200)
    ynew = np.linspace(33.69610595700004, 34.74407959000006, 400)
    if len(lats) == 0:
        r_new = np.zeros((200, 400))
    elif len(lats) > 16:
        newfunc1 = interpolate.interp2d(longs, lats, r_value, kind='cubic')
        r_new = newfunc1(xnew, ynew)
    else:
        lats = lats.repeat(4)
        longs = longs.repeat(4)
        r_value = r_value.repeat(4)
        newfunc1 = interpolate.interp2d(longs, lats, r_value, kind='cubic')
        r_new = newfunc1(xnew, ynew)
    # r_new[r_new < 0] = 0
    # 105.487122  39.585327  111.241882  31.706726
    # float(aa[2]), float(aa[0]), 0.0, float(aa[5]), 0.0, -float(aa[1])
    GeoTransform = [105.487122, (111.241882-105.487122)/float(200), 0.0, 39.585327, 0.0,
                    -(39.585327-31.706726)/float(400)]
    mpl.imshow(r_new, cmap='RdYlGn')
    mpl.colorbar()
    mpl.show()

    xy_number = [200, 400]

    return r_new, GeoTransform, xy_number


def DefinePrj_TmpOut(data_array, GeoTransform, xy_number, tmp_path):
    #定义投影，输出文件tif
    tmp_path_file_aaa = os.path.join(tmp_path, "aaa_test.tif")
    # Prjfl_path = "/Users/shuailee/Documents/ShareDisk/2017.11.10.Restart.Bigdata/Py/py_postgres/test/prj.prj"
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
    OutDat.GetRasterBand(1).SetNoDataValue(-2)
    OutDat.GetRasterBand(1).WriteArray(data_array)
    OutDat = None

'''
def Mask(input_shp_path, out_path, tmp_path):
    #裁剪
    tmp_path_file_aaa = os.path.join(tmp_path, "aaa_test.tif")
    # out_name = "aaa_transform.tif"
    ds = ogr.Open(input_shp_path)
    lyr = ds.GetLayer(0)
    lyr.ResetReading()
    Mask_Commond = 'gdalwarp ' + tmp_path_file_aaa + ' ' + out_path + ' -cutline ' + input_shp_path + ' -dstnodata -1 -crop_to_cutline -overwrite'
    os.system(Mask_Commond)
    os.remove(os.getcwd() + '/aaa_test.tif')
'''


def Mask(input_shp_path, out_path, tmp_path):
    # 裁剪
    tmp_path_file_aaa = os.path.join(tmp_path, "aaa_test.tif")

    cutrst = gdal.Warp(out_path, tmp_path_file_aaa, cutlineDSName=input_shp_path, srcNodata=-2, dstNodata=-2, cropToCutline=True, dstSRS="WGS84")
    cutrst.FlushCache()
    cutrst = None
    print ("Mask OK!")
    os.remove(tmp_path_file_aaa)
    return


def isYear(year):
    # 判断一年中的天数
    if (year%4 == 0) & (year%100 != 0):
        year1 = 366
    elif year % 400 == 0:
        year1 = 366
    else:
        year1 = 365
    return year1


def calcRa(latA, Gsc, doy, J):

    dr = 1 + 0.33 * math.cos((2 * math.pi / doy) * J)
    lat_sun = 0.409 * math.sin((2 * math.pi / 365) * J - 1.39)
    Ws = math.acos(-math.tan((float(latA) / float(180)) * math.pi) * math.tan((float(lat_sun) / float(180)) * math.pi))
    Ra = (24 * (60 / math.pi)) * Gsc * dr * (
    Ws * math.sin((float(latA) / float(180)) * math.pi) * math.sin((float(lat_sun) / float(180)) * math.pi) +
    math.cos((float(latA) / float(180)) * math.pi) * math.cos((float(lat_sun) / float(180)) * math.pi)
    * math.sin((float(Ws) / float(180)) * math.pi))

    return Ra


def calcEa(Ta, RH):

    es = 6.108 * math.exp((17.2 * Ta) / (Ta + 273.3))
    ea = es * RH

    return ea


def calcWs(latA, J):

    lat_sun = 0.409 * math.sin((2 * math.pi / 365) * J - 1.39)
    Ws = math.acos(-math.tan((float(latA) / float(180)) * math.pi) * math.tan((float(lat_sun) / float(180)) * math.pi))

    return Ws


def calcRs(Ra, Ws, As, Bs, n):

    N = float(float(24) / math.pi) * float(Ws)
    Rs = (As + Bs * (n / N)) * Ra

    return Rs


def calcRnl(SBc, Ta, Rs, Ea, As, Bs, Ra):

    Rso = (As + Bs) * Ra
    Rnl = SBc * ((Ta + 273) ** 4) * ((1.35 * (Rs / Rso)) - 0.35) * (0.34 - 0.14 * math.sqrt(Ea))

    return Rnl


def calcRns(Gsc, latA, n, doy, J, Alpha, As, Bs):

    dr = 1 + 0.33 * math.cos((2 * math.pi / doy) * J)
    lat_sun = 0.409 * math.sin((2 * math.pi / 365) * J - 1.39)
    Ws = math.acos(-math.tan((float(latA) / float(180)) * math.pi) * math.tan((float(lat_sun) / float(180)) * math.pi))
    Ra = (24 * (60 / math.pi)) * Gsc * dr * (
    Ws * math.sin((float(latA) / float(180)) * math.pi) * math.sin((float(lat_sun) / float(180)) * math.pi) +
    math.cos((float(latA) / float(180)) * math.pi) * math.cos((float(lat_sun) / float(180)) * math.pi)
    * math.sin((float(Ws) / float(180)) * math.pi))

    N = (24 / math.pi) * Ws
    Rs = (As + Bs * (n / N)) * Ra

    Rns = (1 - Alpha) * Rs
    return Rns


def calcRH(Ta):
    # T = Ta + 273.15
    # RH_real = math.e ** (53.67957 - (6743.769/float(T)) - 4.8451 * math.log(T))
    RH_real = 6.112 ** ((17.67 * Ta)/(Ta + 243.5))
    return RH_real


def calETO(Ta, Ta_1, P, RH, Rn, u12):
    # u2 = u12 * (4.87/math.log(67.8 * z - 5.42))
    u2 = 0.72 * u12
    gam = 2.501 - (0.002361 * Ta)
    Gam = 0.0016286 * (P / gam)
    es = 6.108 * math.exp((17.2 * Ta) / (Ta + 273.3))
    ea = es * RH
    svpd = (4098 * ea) / ((Ta + 273.3) ** 2)
    G = 0.38 * (Ta - Ta_1)
    ET0 = (0.408 * (Rn - G) + Gam * (float(900) / float(Ta + 273)) * u2 * float(es - ea)) / float(svpd + Gam * (1 + 0.34 * u2))

    return ET0


def calcR(code, lon, lat, time_data, tem_mean, prp_sum, sun_sum, wind_mean, rehum_mean, p_mean, start_time, end_time):
    #根据Penman公式计算ET0, ET0/P计算干旱指数r
    '''
    As, Bs为回归系数，表达地外辐射到达地球的比例regression constant, expressing the fraction of extraterrestrial radiation reaching the earth,
	As：0.1852222
	Bs：0.555
    n：actual duration of sunshine (hour)
    N：maximum possible duration of sunshine or daylight hours (hour).
    Ws：日落时角（弧度制） sunset hour angle[rad].
    Ra：extraterrestrial radiation (MJ m-2 day-1).

    Gsc：太阳常数 solar constant (MJ m-2 min-1),
    dr：反向日地距离 inverse relative distance Earth-Sun,
    latA：纬度（弧度制） latitude[rad],
    lat_sun：太阳赤纬（弧度制） solar declination
    Rso：晴空太阳辐射 clear-sky solar radiation (MJ m-2 day-1)
    SBc：斯蒂芬-波茨曼常数 Stefan-Boltzmann constant [4.903 10-9 MJ K-4 m-2 day-1]

    G：土壤热通量密度（MJ m-2 day-1） soil heat flux (MJ m-2 day-1)，
    Ta：第day天的平均气温(°C)，
    svpd：饱和水气压与气温曲线斜率 slope vapour pressure deficit (kPa °C-1)，
    es：饱和水气压 (kPa)，
    ea：实际水汽压 (kPa)，
    RH：相对湿度
    pc：干湿表常数 psychrometric constant (kPa °C-1)，
    P：空气压(kPa)，
    Gam：气化潜热(MJ kg-1),
    u2：地表面以上2m处风速(m s-1)，
    u12：在z m处实际观测风速(m s-1)，VI中z取值12m

    '''

    no_value = -9999
    start_time = start_time.split("-")
    end_time = end_time.split("-")

    As = 0.1852222
    Bs = 0.555
    Alpha = 0.23
    Gsc = 0.0820
    SBc = 4.903 * 10**-9


    start_time_0 = datetime.date(int(start_time[0]), int(start_time[1]), int(start_time[2]))
    J = start_time_0.timetuple().tm_yday
    doy = isYear(int(start_time[0]))

    time_begin = datetime.date(int(start_time[0]), int(start_time[1]), int(start_time[2]))
    time_end = datetime.date(int(end_time[0]), int(end_time[1]), int(end_time[2]))

    cal_day = time_begin
    delta = datetime.timedelta(days=1)

    lons = []
    lats = []
    ET0s = []
    Prps = []
    codes = []
    # 逐站点、逐天计算
    while cal_day <= time_end:
        cal_day_1 = (cal_day - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        cc = [i for i, a in enumerate(time_data) if (a == cal_day.strftime("%Y-%m-%d"))]
        cc_1 = [i for i, a in enumerate(time_data) if (a == cal_day_1)]

        code_0 = [code[i] for i in cc]
        code_1 = [code[i] for i in cc_1]
        lon_0 = [lon[i] for i in cc]
        lat_0 = [lat[i] for i in cc]

        tem_mean_0 = [tem_mean[i] for i in cc]
        tem_mean_1 = [tem_mean[i] for i in cc_1]
        prp_sum_0 = [prp_sum[i] for i in cc]
        sun_sum_0 = [sun_sum[i] for i in cc]
        wind_0 = [wind_mean[i] for i in cc]
        rehum_0 = [rehum_mean[i] for i in cc]
        p_mean_0 = [p_mean[i] for i in cc]


        for i in range(0, len(code_0)):
            ccc = [ii for ii, a in enumerate(code_1) if a == code_0[i]]
            if ((tem_mean_0[i] == no_value) | (prp_sum_0[i] == no_value) | (sun_sum_0[i] == no_value) | (wind_0[i] == no_value) | \
                    (rehum_0[i] == no_value) | (p_mean_0[i] == no_value)):
                continue
            else:
                # 经纬度

                latA = lat_0[i]
                n = sun_sum_0[i]
                Ta = tem_mean_0[i]
                P = p_mean_0[i]
                RH = rehum_0[i]
                Ta_1 = tem_mean_1[ccc[0]]
                Prp = prp_sum_0[i]
                u12 = wind_0[i]

                Ra = calcRa(latA, Gsc, doy, J)
                Ws = calcWs(latA, J)
                Rs = calcRs(Ra, Ws, As, Bs, n)
                Ea = calcEa(Ta, RH)

                Rns = calcRns(Gsc, latA, n, doy, J, Alpha, As, Bs)
                Rnl = calcRnl(SBc, Ta, Rs, Ea, As, Bs, Ra)

                Rn = Rns - Rnl

                # RH_real = calcRH(Ta)
                ET0 = calETO(Ta, Ta_1, P, RH, Rn, u12)

                ET0s.append(ET0)
                Prps.append(Prp)
                lons.append(lon_0[i])
                lats.append(lat_0[i])
                codes.append(code_0[i])
        cal_day += delta

    # 逐站点累和
    site_code = list(set(codes))
    site_code.sort(key=codes.index)

    site_lons = []
    site_lats = []
    site_r = []
    for i in range(len(site_code)):
        site_code_tmp = site_code[i]
        ccc = [i for i, a in enumerate(codes) if a == site_code_tmp]

        site_lons.append(lons[ccc[0]])
        site_lats.append(lats[ccc[0]])
        site_pre = [Prps[i] for i in ccc]
        site_et0 = [ET0s[i] for i in ccc]
        if np.nansum(np.array(site_pre)) <> 0:
            site_r.append(np.nansum(np.array(site_et0))/np.nansum(np.array(site_pre)))
        else:
            site_r.append(np.nan)

    return site_lons, site_lats, site_r


def Read_Check(path, start_time, end_time):
    #读取excel, 提取相应时间段的各要素数据
    # start_time = start_time.split("-")
    # new_start_time =
    no_value = -9999

    start_time_1 = (datetime.date(int(start_time.split("-")[0]), int(start_time.split("-")[1]),
                        int(start_time.split("-")[2])) - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        all_data = xlrd.open_workbook(path)
    except:
        print "File %s isn't exist, please check" % path

    table = all_data.sheets()[0]

    code = table.col_values(0)[1:]
    lon_data = table.col_values(2)[1:]
    lat_data = table.col_values(3)[1:]
    time_data = [datetime.datetime(*xldate_as_tuple(i, 0)).strftime('%Y-%m-%d') for i in table.col_values(4)[1:]]
    # time_data = table.col_values(4)[1:]
    tem_mean = table.col_values(5)[1:]
    prp_sum = table.col_values(6)[1:]
    sun_sum = table.col_values(7)[1:]
    wind_mean = table.col_values(8)[1:]
    rehum_mean = table.col_values(9)[1:]
    p_mean = table.col_values(10)[1:]

    # cc = [i for i, a in enumerate(time_data) if (a == start_time) | (a == aa)]
    cc = [i for i, a in enumerate(time_data) if (a >= start_time_1) & (a <= end_time )]

    code_1 = [int(code[i]) for i in cc]
    lon_1 = [float(lon_data[i]) for i in cc]
    lat_1 = [float(lat_data[i]) for i in cc]
    time_data_1 = [time_data[i] for i in cc]

    tem_mean_1 = [float(tem_mean[i]) for i in cc]
    prp_sum_1 = [float(prp_sum[i]) for i in cc]
    sun_sum_1 = [float(sun_sum[i]) for i in cc]
    wind_1 = [float(wind_mean[i]) for i in cc]
    rehum_1 = [float(rehum_mean[i]) for i in cc]
    p_mean_1 = [float(p_mean[i]) for i in cc]

    # dd = [i for i in enumerate(len(tem_mean_1)) if tem_mean_1[i] != no_value & ]
    dd = []
    for i in range(len(tem_mean_1)):
        if (tem_mean_1[i] != no_value) & (prp_sum_1[i] != no_value) & (sun_sum_1[i] != no_value) & (wind_1[i] != no_value) & (rehum_1[i] != no_value) & (p_mean_1[i] != no_value):
            dd.append(i)

    code_1 = [int(code_1[i]) for i in dd]
    lon_1 = [float(lon_1[i]) for i in dd]
    lat_1 = [float(lat_1[i]) for i in dd]
    time_data_1 = [time_data_1[i] for i in dd]

    tem_mean_1 = [float(tem_mean_1[i]) for i in dd]
    prp_sum_1 = [float(prp_sum_1[i]) for i in dd]
    sun_sum_1 = [float(sun_sum_1[i]) for i in dd]
    wind_1 = [float(wind_1[i]) for i in dd]
    rehum_1 = [float(rehum_1[i]) for i in dd]
    p_mean_1 = [float(p_mean_1[i]) for i in dd]

    return code_1, lon_1, lat_1, time_data_1, tem_mean_1, prp_sum_1, sun_sum_1, wind_1, rehum_1, p_mean_1


def main(path, shp_path, out_path, tmp_path, start_time, end_time):
    # 读取数据
    code, lon, lat, time_data, tem_mean, prp_sum, sun_sum, wind, rehum, p_mean = Read_Check(path, start_time, end_time)

    # 计算干旱指数
    longs, lats, r_value = calcR(code, lon, lat, time_data, tem_mean, prp_sum, sun_sum, wind, rehum, p_mean, start_time, end_time)

    # 插值操作与临时保存
    interpolation(shp_path, out_path, tmp_path, longs, lats, r_value)
    # r_new, GeoTransform, xy_number = interpolation(longs, lats, r_value)
    # print ("60%")
    # # 定义投影、裁剪与保存
    # DefinePrj_TmpOut(r_new, GeoTransform, xy_number, tmp_path)
    # print ("80%")
    # Mask(shp_path, out_path, tmp_path)
    # print ("100%")


    return

if __name__ == '__main__':
    path = "/Users/shuailee/Documents/Climate/Example/2017-06-01-2017-06-30.xlsx"
    shp_path = "/Users/shuailee/Documents/Climate/Example/Boundary/shaanxi.shp"
    out_path = "/Users/shuailee/Documents/Climate/tmp"
    tmp_path = "/Users/shuailee/Documents/Climate/tmp"
    out_name = "shaanxi_penman_test.tif"
    start_time = "2017-06-01"
    end_time = "2017-06-30"

    if not os.path.exists(out_path):
        ccc = "mkdir %s" % out_path
        os.system(ccc)

    out_path_file = os.path.join(out_path, out_name)
    main(path, shp_path, out_path_file, tmp_path, start_time, end_time)