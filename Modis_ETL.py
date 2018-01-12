# -*- encoding: utf-8 -*-

import sys
import os
from osgeo import ogr
import osr
import gdal
import matplotlib.pyplot as mpl
import numpy as np
import datetime
import natsort
# %matplotlib inline

def LeaporNot(year):
    if (year%4 == 0) & (year%100 != 0):
        year1 = 366
    elif year % 400 == 0:
        year1 = 366
    else:
        year1 = 365
    return year1


def month2day(start_time, end_time):
    start_time = start_time.split("-")
    end_time = end_time.split("-")
    start_day = datetime.date(int(start_time[0]), int(start_time[1]), int(start_time[2]))
    end_day = datetime.date(int(end_time[0]), int(end_time[1]), int(end_time[2]))
    year_day = [int(start_time[0]), start_day.timetuple().tm_yday,
                int(end_time[0]), end_day.timetuple().tm_yday]
    return year_day


def Get_Layer(ss, name):
    aa = []
    for sd, des in ss.GetSubDatasets():
        # print sd
        if sd.endswith(name):
            aa = sd
    return aa


def Mosaic(input_path1, input_path2, name):
    #镶嵌，左右镶嵌
    # read hdf raster
    ss1 = gdal.Open(input_path1)
    ss2 = gdal.Open(input_path2)
    layer_name1 = Get_Layer(ss1, name)
    layer_name2 = Get_Layer(ss2, name)
    # print(layer_name)
    if layer_name1 is None:
        sys.exit("Could not open {0}".format(input_path1))
    else:
        dssub1 = gdal.Open(layer_name1)
        data1 = dssub1.ReadAsArray().astype(np.float)
    try:
        dssub2 = gdal.Open(layer_name2)
        data2 = dssub2.ReadAsArray().astype(np.float)
    except:
        sys.exit("Could not open {0}".format(input_path2))

    data4 = np.concatenate((data1, data2), axis=1)
    data4 = data4 / 10000
    data4[data4 < 0] = -1
    return data4


def Transform(out_path):
    #转投影
    Prjfl_path = "/Users/shuailee/Documents/ShareDisk/2017.11.10.Restart.Bigdata/Py/py_postgres/test/prj.prj"
    aaa = os.getcwd() + '/aaa_transform.tif'
    RePrjStr = 'gdalwarp ' + aaa + ' ' + out_path + ' -t_srs ' + Prjfl_path + ' -srcnodata -2'
    print RePrjStr
    os.system(RePrjStr)
    print ("Transform ok!")
    os.remove(aaa)


def Mask(input_shp_path):
    #裁剪
    aaa = os.getcwd() + '/aaa_test.tif'
    out_name = "aaa_transform.tif"
    ds = ogr.Open(input_shp_path)
    lyr = ds.GetLayer(0)
    lyr.ResetReading()
    Mask_Commond = 'gdalwarp ' + aaa + ' ' + out_name + ' -cutline ' + input_shp_path + ' -dstnodata -1 -crop_to_cutline -overwrite'
    os.system(Mask_Commond)
    os.remove(os.getcwd() + '/aaa_test.tif')


def dirlist(dir_path, allfile, data_format):
    filelist = os.listdir(dir_path)
    for filename in filelist:
        filepath = os.path.join(dir_path, filename)
        if os.path.isdir(filepath):
            dirlist(filepath, allfile, data_format)
        else:
            if filepath.endswith("." + data_format):
                allfile.append(filepath)

    allfile = natsort.natsorted(allfile)
    return allfile


def YearNumH_check(year_list, num_list, h_list, year_day):
    #取出满足条件的年、天、h
    start_file_h26_c = []
    start_file_h27_c = []
    end_file_h26_c = []
    end_file_h27_c = []
    for i in range(len(year_list)):
        if (year_list[i] == int(year_day[0])-1) & (num_list[i] >= year_day[1]) & (num_list[i] <= year_day[3]) & (h_list[i] == 26):
            start_file_h26_c.append(i)
        elif (year_list[i] == int(year_day[0])-1) & (num_list[i] >= year_day[1]) & (num_list[i] <= year_day[3]) & (h_list[i] == 27):
            start_file_h27_c.append(i)
        elif (year_list[i] == int(year_day[0])) & (num_list[i] >= year_day[1]) & (num_list[i] <= year_day[3]) & (h_list[i] == 26):
            end_file_h26_c.append(i)
        elif (year_list[i] == int(year_day[0])) & (num_list[i] >= year_day[1]) & (num_list[i] <= year_day[3]) & (h_list[i] == 27):
            end_file_h27_c.append(i)

    return start_file_h26_c, start_file_h27_c, end_file_h26_c, end_file_h27_c


def Check_Take(path, file_list, year_day):
    #取出此时间段的所有hdf数据集
    #MOD13Q1.A2016001.h26v05.006.2016029065706
    mod_day = [g for g in range(1, 365, 16)]
    mod_day_check = list(filter(lambda i: (i > year_day[1]) & (i < year_day[3]), mod_day))
    start_file_h26 = []
    start_file_h27 = []
    end_file_h26 = []
    end_file_h27 = []
    aa = [".".join(os.path.basename(i).split(".")[0:-3]) for i in file_list]
    # aa = [".".join(i.split("/")[-1].split(".")[0:-3]) for i in file_list]
    for i in range(len(mod_day_check)):
        start_MOD_name_h26 = "MOD13Q1.A%s%03d.h%sv05" % (year_day[0]-1, mod_day_check[i], "26")
        start_MOD_name_h27 = "MOD13Q1.A%s%03d.h%sv05" % (year_day[0]-1, mod_day_check[i], "27")
        end_MOD_name_h26 = "MOD13Q1.A%s%03d.h%sv05" % (year_day[0], mod_day_check[i], "26")
        end_MOD_name_h27 = "MOD13Q1.A%s%03d.h%sv05" % (year_day[0], mod_day_check[i], "27")
        try:
            cc = aa.index(start_MOD_name_h26) & aa.index(start_MOD_name_h27) & aa.index(end_MOD_name_h26) & aa.index(end_MOD_name_h27)
            start_file_h26.append(file_list[aa.index(start_MOD_name_h26)])
            start_file_h27.append(file_list[aa.index(start_MOD_name_h27)])
            end_file_h26.append(file_list[aa.index(end_MOD_name_h26)])
            end_file_h27.append(file_list[aa.index(end_MOD_name_h27)])
        except:
            print ("The file %s isn't exit.please check!" % start_MOD_name_h26)

    start_file_h26 = natsort.natsorted(start_file_h26)
    start_file_h27 = natsort.natsorted(start_file_h27)
    end_file_h26 = natsort.natsorted(end_file_h26)
    end_file_h27 = natsort.natsorted(end_file_h27)
    return start_file_h26, start_file_h27, end_file_h26, end_file_h27


def MVC_Mosaic(start_file_h26, start_file_h27, end_file_h26, end_file_h27):
    #镶嵌和MVC
    name = "250m 16 days NDVI"
    aaa = "aaa_test.tif"

    start_data_1 = Mosaic(start_file_h26[0], start_file_h27[0], name)
    end_data_1 = Mosaic(end_file_h26[0], end_file_h27[0], name)
    start_data = np.zeros((start_data_1.shape[0], start_data_1.shape[1], len(start_file_h26)))
    end_data = np.zeros((start_data_1.shape[0], start_data_1.shape[1], len(start_file_h26)))

    start_data[:, :, 0] = start_data_1
    end_data[:, :, 0] = end_data_1
    for i in range(1, min(len(start_file_h26), len(start_file_h27), len(end_file_h26), len(end_file_h27))):
        print i
        start_data_tmp = Mosaic(start_file_h26[i], start_file_h27[i], name)
        end_data_tmp = Mosaic(end_file_h26[i], end_file_h27[i], name)
        start_data[:, :, i] = start_data_tmp
        end_data[:, :, i] = end_data_tmp
    start_data_mvc = start_data.max(axis=2)
    end_data_mvc = end_data.max(axis=2)
    start_data_mvc[start_data_mvc == -1] = None
    end_data_mvc[end_data_mvc == -1] = None
    last_data = end_data_mvc - start_data_mvc
    last_data[np.isnan(last_data)] = -2

    #读取一个地理参考和投影，将last_data保存为临时文件
    ss1 = gdal.Open(start_file_h26[0])
    layer_name1 = Get_Layer(ss1, name)
    dssub1 = gdal.Open(layer_name1)
    OutDrv = gdal.GetDriverByName('GTiff')
    OutDat = OutDrv.Create(aaa,
                           dssub1.RasterXSize * 2, dssub1.RasterYSize, 1,
                           gdal.GDT_Float64, ["COMPRESS=DEFLATE"])
    assert OutDat is not None
    OutDat.SetGeoTransform(dssub1.GetGeoTransform())
    aa = dssub1.GetProjection()
    OutDat.SetProjection(dssub1.GetProjection())
    OutDat.GetRasterBand(1).WriteArray(last_data)
    OutDat = None


def main(path, shp_path, out_path, start_time, end_time):
    #查找
    year_day = month2day(start_time, end_time)
    file_list = dirlist(path, [], "hdf")
    start_file_h26, start_file_h27, end_file_h26, end_file_h27 = Check_Take(path, file_list, year_day)

    #分别拼接+MVC+差值
    MVC_Mosaic(start_file_h26, start_file_h27, end_file_h26, end_file_h27)

    #镶嵌+裁剪+转投影
    Mask(shp_path)
    Transform(out_path)


if __name__ == '__main__':

    path = "/Users/shuailee/Documents/Climate/Example/MOD13Q1"
    shp_path = "/Users/shuailee/Documents/ShareDisk/2017.11.10.Restart.Bigdata/programme/data/xian_boundry/xian.shp"
    out_path = os.getcwd() + "/xian_caijian.tif"
    start_time = "2017-10-02"
    end_time = "2017-12-14"

    main(path, shp_path, out_path, start_time, end_time)