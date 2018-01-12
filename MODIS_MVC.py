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


'''
def Transform(tmp_path):
    #转投影
    Prjfl_path = "/Users/shuailee/Documents/ShareDisk/2017.11.10.Restart.Bigdata/Py/py_postgres/test/prj.prj"
    tmp_path_file_aaa = os.path.join(tmp_path, "aaa_test.tif")
    tmp_path_file_trans = os.path.join(tmp_path, "aaa_transform.tif")
    RePrjStr = 'gdalwarp ' + tmp_path_file_aaa + ' ' + tmp_path_file_trans + ' -t_srs ' + Prjfl_path + ' -srcnodata -2 -dstnodata -2 -multi'
    os.system(RePrjStr)
    print ("Transform ok!")
    os.remove(tmp_path_file_aaa)


def Mask(input_shp_path, out_path, tmp_path):
    #裁剪
    tmp_path_file_trans = os.path.join(tmp_path, "aaa_transform.tif")
    ds = ogr.Open(input_shp_path)
    lyr = ds.GetLayer(0)
    lyr.ResetReading()
    Mask_Commond = 'gdalwarp ' + tmp_path_file_trans + ' ' + out_path + ' -cutline ' + input_shp_path + ' -srcnodata -2 -dstnodata -2 -crop_to_cutline -overwrite -multi'
    os.system(Mask_Commond)
    os.remove(tmp_path_file_trans)
'''


def Mask_Transform(input_shp_path, out_path, tmp_path):
    # 裁剪转投影
    Prjfl_path = "/Users/shuailee/Documents/ShareDisk/2017.11.10.Restart.Bigdata/Py/py_postgres/test/prj.prj"
    tmp_path_file_aaa = os.path.join(tmp_path, "aaa_test.tif")
    # tmp_path_file_trans = os.path.join(tmp_path, "aaa_transform.tif")

    cutrst = gdal.Warp(out_path, tmp_path_file_aaa, cutlineDSName=input_shp_path, srcNodata=-2, dstNodata=-2, cropToCutline=True, dstSRS="WGS84")
    cutrst.FlushCache()
    cutrst = None
    print ("Mask OK!")
    return


def Read_Tiff(file):
    dssub = gdal.Open(file)
    data_tmp = dssub.ReadAsArray().astype(np.float)
    data_tmp[data_tmp < 0] = None
    return data_tmp


def MVC_Difference(start_files, end_files, tmp_path):
    #读取, MVC, 差值
    tmp_path_file = os.path.join(tmp_path, "aaa_test.tif")
    if len(start_files) == len(end_files):

        start_data_tmp = Read_Tiff(start_files[0])

        end_data_tmp = Read_Tiff(end_files[0])

        start_data = np.zeros((start_data_tmp.shape[0], start_data_tmp.shape[1], len(start_files)))
        end_data = np.zeros((start_data_tmp.shape[0], start_data_tmp.shape[1], len(start_files)))
        start_data[:, :, 0] = start_data_tmp
        end_data[:, :, 0] = end_data_tmp

        for i in range(1, len(start_files)):
            start_data_tmp1 = Read_Tiff(start_files[i])
            end_data_tmp1 = Read_Tiff(end_files[i])

            start_data[:, :, i] = start_data_tmp1
            end_data[:, :, i] = end_data_tmp1
        start_data = start_data.max(axis=2)

        # for i in range(1, len(start_files)):
        #     end_data_tmp1 = Read_Tiff(end_files[i])
        #     end_data[:, :, i] = end_data_tmp1
        # end_data = end_data.max(axis=2)

        Diff_data = end_data - start_data
        Diff_data[np.isnan(Diff_data)] = -2

        dssub1 = gdal.Open(start_files[0])
        OutDrv = gdal.GetDriverByName('GTiff')
        OutDat = OutDrv.Create(tmp_path_file,
                               dssub1.RasterXSize, dssub1.RasterYSize, 1,
                               gdal.GDT_Float64, ["COMPRESS=DEFLATE"])
        assert OutDat is not None
        OutDat.SetGeoTransform(dssub1.GetGeoTransform())
        OutDat.SetProjection(dssub1.GetProjection())
        OutDat.GetRasterBand(1).WriteArray(Diff_data)
        OutDat = None

    else:
        print ("please check length(start_files) <> length(end_files)")


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


def Check_Take(start_year_file_list, end_year_file_list, year_day):
    #取出此时间段的所有hdf数据集
    #MOD13Q1.A2016001.h26v05.006.2016029065706
    mod_day = [g for g in range(1, 365, 16)]
    mod_day_check = list(filter(lambda i: (i > year_day[1]) & (i < year_day[3]), mod_day))

    start_files = []
    end_files = []

    start_aa = [".".join(os.path.basename(i).split(".")[0:2]) for i in start_year_file_list]
    end_aa = [".".join(os.path.basename(i).split(".")[0:2]) for i in end_year_file_list]
    # aa = [".".join(i.split("/")[-1].split(".")[0:-3]) for i in file_list]

    for i in range(len(mod_day_check)):
        start_MOD_name = "MOD13Q1.A%s%03d" % (year_day[0]-1, mod_day_check[i])
        end_MOD_name = "MOD13Q1.A%s%03d" % (year_day[0], mod_day_check[i])
        try:
            cc = (start_aa.index(start_MOD_name)) & (end_aa.index(end_MOD_name))
            start_files.append(start_year_file_list[start_aa.index(start_MOD_name)])

            end_files.append(end_year_file_list[end_aa.index(end_MOD_name)])

        except:
            print ("The file %s or %s isn't exit.please check!" % (start_MOD_name, end_MOD_name))

    start_files = natsort.natsorted(start_files)
    end_files = natsort.natsorted(end_files)
    return start_files, end_files


def main(path, out_path, shp_path, start_time, end_time, tmp_path):
    # 查找
    year_day = month2day(start_time, end_time)
    start_year_path = os.path.join(path, str(int(start_time[0:4]) - 1))
    end_year_path = os.path.join(path, str(int(start_time[0:4])))
    start_year_file_list = dirlist(start_year_path, [], "tif")
    end_year_file_list = dirlist(end_year_path, [], "tif")
    start_files, end_files = Check_Take(start_year_file_list, end_year_file_list, year_day)

    # MVC+差值
    MVC_Difference(start_files, end_files, tmp_path)

    # 裁剪+转投影
    Mask_Transform(shp_path, out_path, tmp_path)
    # Transform(tmp_path)
    # Mask(shp_path, out_path, tmp_path)




if __name__ == '__main__':

    path = "/Users/shuailee/Documents/Climate/test"
    out_path = "/Users/shuailee/Documents/Climate/tmp/MODIS_xian.tif"
    shp_path = "/Users/shuailee/Documents/ShareDisk/2017.11.10.Restart.Bigdata/programme/data/xian_boundry/xian.shp"
    tmp_path = "/Users/shuailee/Documents/Climate/tmp"
    start_time = "2017-1-1"
    end_time = "2017-12-30"

    main(path, out_path, shp_path, start_time, end_time, tmp_path)