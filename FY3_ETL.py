# -*- encoding: utf-8 -*-

# import sys
import os
# from osgeo import ogr
# import osr
import gdal
# import matplotlib.pyplot as mpl
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import datetime
import natsort
# %matplotlib inline



def calcNDVI(input_fn):
    dat = gdal.Open(input_fn)
    band1 = dat.GetRasterBand(1)
    band2 = dat.GetRasterBand(2)
    RED = band1.ReadAsArray().astype(np.float32)
    NIR = band2.ReadAsArray().astype(np.float32)

    NDVI = (NIR - RED) / (NIR + RED)

    NDVI[NDVI < 0] = -1
    # OutDrv = gdal.GetDriverByName('GTiff')
    # OutDat = OutDrv.Create(outfn,
    #                           dat.RasterXSize, dat.RasterYSize, 1,
    #                           gdal.GDT_Float32, ["COMPRESS=DEFLATE"])
    # assert OutDat is not None
    # OutDat.SetGeoTransform(dat.GetGeoTransform())
    # print dat.GetGeoTransform()
    # OutDat.SetProjection(dat.GetProjection())
    # print dat.GetProjection()
    # OutDat.GetRasterBand(1).SetNoDataValue(-1)
    # OutDat.GetRasterBand(1).WriteArray(NDVI)
    # OutDat = None
    return NDVI


def MVC_read(start_files, end_files, tmp_path):
    #读取数据，计算NDVI，MVC，差值

    tmp_path_file_aaa = os.path.join(tmp_path, "aaa_test.tif")
    start_data_tmp = calcNDVI(start_files[0])
    end_data_tmp = calcNDVI(end_files[0])
    start_data = np.zeros((start_data_tmp.shape[0], start_data_tmp.shape[1], len(start_files)))
    end_data = np.zeros((start_data_tmp.shape[0], start_data_tmp.shape[1], len(start_files)))

    start_data[:, :, 0] = start_data_tmp
    end_data[:, :, 0] = end_data_tmp
    for i in range(1, min(len(start_files), len(end_files))):
        start_data_tmp1 = calcNDVI(start_files[i])
        start_data_tmp1[np.isnan(start_data_tmp1)] = -1
        end_data_tmp1 = calcNDVI(end_files[i])
        end_data_tmp1[np.isnan(end_data_tmp1)] = -1
        if (start_data_tmp1.shape[0] == start_data_tmp.shape[0]) & (start_data_tmp1.shape[1] == start_data_tmp.shape[1]):
            start_data[:, :, i] = start_data_tmp1
            end_data[:, :, i] = end_data_tmp1
        else:
            continue

    start_data = start_data.max(axis=2)
    end_data = end_data.max(axis=2)
    start_data[start_data == -1] = None
    end_data[end_data == -1] = None
    last_data = end_data - start_data
    last_data[np.isnan(last_data)] = -2

    # 读取一个地理参考和投影，将last_data保存为临时文件
    dssub1 = gdal.Open(start_files[0])
    OutDrv = gdal.GetDriverByName('GTiff')
    OutDat = OutDrv.Create(tmp_path_file_aaa,
                           dssub1.RasterXSize, dssub1.RasterYSize, 1,
                           gdal.GDT_Float64, ["COMPRESS=DEFLATE"])
    assert OutDat is not None
    OutDat.SetGeoTransform(dssub1.GetGeoTransform())
    OutDat.SetProjection(dssub1.GetProjection())
    # OutDat.GetRasterBand(1).SetNoDataValue(-1)
    OutDat.GetRasterBand(1).WriteArray(last_data)
    OutDat = None


'''
def Transform(out_path):
    #转投影
    Prjfl_path = "/Users/shuailee/Documents/ShareDisk/2017.11.10.Restart.Bigdata/Py/py_postgres/test/prj.prj"
    aaa = os.getcwd() + '/aaa_transform.tif'
    RePrjStr = 'gdalwarp ' + aaa + ' ' + out_path + ' -t_srs ' + Prjfl_path + ' -srcnodata -1'
    os.system(RePrjStr)
    print "Transform ok!"
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
'''


def Mask_Transform(input_shp_path, out_path, tmp_path):
    # 裁剪转投影
    # Prjfl_path = "/Users/shuailee/Documents/ShareDisk/2017.11.10.Restart.Bigdata/Py/py_postgres/test/prj.prj"
    tmp_path_file_aaa = os.path.join(tmp_path, "aaa_test.tif")
    # tmp_path_file_trans = os.path.join(tmp_path, "aaa_transform.tif")

    cutrst = gdal.Warp(out_path, tmp_path_file_aaa, cutlineDSName=input_shp_path, srcNodata=-2, dstNodata=-2, cropToCutline=True, dstSRS="WGS84")
    cutrst.FlushCache()
    cutrst = None
    print ("Mask OK!")
    os.remove(tmp_path_file_aaa)
    return


def Check_file(file_list, start_time, end_time, Star_num):
    #提取整个时间段内的文件，若不存在跳过

    start_year_time1 = datetime.date(int(start_time.split("-")[0])-1, int(start_time.split("-")[1]), int(start_time.split("-")[2]))
    start_year_time2 = datetime.date(int(end_time.split("-")[0])-1, int(end_time.split("-")[1]), int(end_time.split("-")[2]))

    start_files = []
    end_files = []

    d = start_year_time1
    delta = datetime.timedelta(days=1)
    aa = ["_".join(os.path.basename(i).split("_")[0:-2]) for i in file_list]
    # aa = ["_".join(i.split("/")[-1].split("_")[0:-2]) for i in file_list]
    # print aa[1]
    while d <= start_year_time2:
        new_day = d.strftime("%Y-%m-%d")
        start_time_tmp = "".join([str(int(new_day.split("-")[0])), new_day.split("-")[1], new_day.split("-")[2]])
        end_time_tmp = "".join([str(int(new_day.split("-")[0])+1), new_day.split("-")[1], new_day.split("-")[2]])
        start_FY3_name = "%s_VIRR_DXX_GLL_L1_%s" % (Star_num, start_time_tmp)
        end_FY3_name = "%s_VIRR_DXX_GLL_L1_%s" % (Star_num, end_time_tmp)
        try:
            print
            cc = (aa.index(start_FY3_name)) & (aa.index(end_FY3_name))
            start_files.append(file_list[aa.index(start_FY3_name)])
            end_files.append(file_list[aa.index(end_FY3_name)])
        except:
            print "The file %s or %s isn't exit.please check!" % (start_FY3_name, end_FY3_name)
        d += delta

    return start_files, end_files


def Check_Folder(path, Star_num, start_time, end_time):
    #查找出相关的两个文件夹

    start_time = "%s%2d" % (str(int(start_time.split("-")[0]) - 1), int(start_time.split("-")[1]))
    end_time = "%s%2d" % (str(int(end_time.split("-")[0])), int(end_time.split("-")[1]))
    start_folder = "%s/%s/%s" % (path, start_time, Star_num)
    end_folder = "%s/%s/%s" % (path, end_time, Star_num)
    return start_folder, end_folder


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


def main(path, shp_path, out_path, start_time, end_time, Star_num, tmp_path):

    #check files
    file_list = dirlist(path, [], "LDF")
    start_files, end_files = Check_file(file_list, start_time, end_time, Star_num)

    #Read, Calculate NDVI, MVC, Subtract
    MVC_read(start_files, end_files, tmp_path)

    #Transform Project, Mask
    Mask_Transform(shp_path, out_path, tmp_path)
    # Mask(shp_path)
    # Transform(out_path)


if __name__ == '__main__':
    # FY3C_VIRR_DXX_GLL_L1_20170309_1113_1000M.LDF
    # FY3B_VIRR_DXX_GLL_L1_20160213_1516_1000M.LDF
    path = "/Users/shuailee/Documents/Climate/Example/FY3"
    shp_path = "/Users/shuailee/Documents/Climate/Example/Boundary/shaanxi.shp"
    out_path = "/Users/shuailee/Documents/Climate/tmp"
    out_name = "Shaanxi_caijian.tif"
    tmp_path = "/Users/shuailee/Documents/Climate/tmp"
    start_time = "2017-1-02"
    end_time = "2017-3-30"
    Star_num = "FY3B"

    out_path_file = os.path.join(out_path, out_name)

    main(path, shp_path, out_path_file, start_time, end_time, Star_num, tmp_path)