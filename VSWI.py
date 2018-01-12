# -*- encoding: utf-8 -*-

# import sys
import os
# from osgeo import ogr
# import osr
import gdal
import matplotlib.pyplot as mpl
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import datetime
import natsort
# %matplotlib inline


def calcNDVI(input_fn):
    #Calculate NDVI
    dat = gdal.Open(input_fn)
    band1 = dat.GetRasterBand(1)
    band2 = dat.GetRasterBand(2)
    RED = band1.ReadAsArray().astype(np.float32)
    NIR = band2.ReadAsArray().astype(np.float32)

    NDVI = (NIR - RED) / (NIR + RED)

    NDVI[NDVI < 0] = -1
    return NDVI


def calcPv(NDVI):
    #Calculate Pv
    Imax = 0.5
    Imin = 0.2
    Pv = np.square((NDVI - Imin)/(Imax - Imin))
    return Pv


def calcLST(input_fn, NDVI):
    #Calculate LST
    m_4 = 0.0107
    n_4 = 0.9793
    m_5 = 0.0030
    n_5 = 0.9870
    Imax = 0.5
    Imin = 0.2

    dat = gdal.Open(input_fn)
    band1 = dat.GetRasterBand(3)
    band2 = dat.GetRasterBand(4)
    band_4 = band1.ReadAsArray().astype(np.float32)
    band_5 = band2.ReadAsArray().astype(np.float32)

    # NDVI = calcNDVI(input_fn)

    Pv = np.square((NDVI - Imin) / (Imax - Imin))
    e_4 = m_4 * Pv + n_4
    e_5 = m_5 * Pv + n_5
    T_e = e_4 - e_5
    ee = (e_4 + e_5) / 2

    P = 1 + (0.1197*(1-ee)/ee) - (0.4891*T_e/np.square(ee))
    M = 5.6538 + (5.6543*(1-ee)/ee) - (12.9238*T_e/np.square(ee))

    LST = (P * (band_4 + band_5))/2 + (M * (band_4 - band_5))/2 - 0.14

    return LST


def Mean_LST_MVC_NDVI(files, tmp_path):
    # 读取数据，计算NDVI MVC，LST mean, VSWI

    tmp_path_file_aaa = os.path.join(tmp_path, "aaa_test.tif")
    NDVI_0 = calcNDVI(files[0])
    LST_0 = calcLST(files[0], NDVI_0)
    LST_all = np.zeros((NDVI_0.shape[0], NDVI_0.shape[1], len(files)))
    NDVI_all = np.zeros((LST_0.shape[0], LST_0.shape[1], len(files)))

    LST_all[:, :, 0] = LST_0
    NDVI_all[:, :, 0] = NDVI_0
    for i in range(1, len(files)):
        NDVI_tmp = calcNDVI(files[i])
        LST_tmp = calcLST(files[i], NDVI_tmp)
        LST_all[:, :, i] = LST_tmp
        NDVI_all[:, :, i] = NDVI_tmp
    NDVI_mvc = np.nanmax(NDVI_all, axis=2)
    NDVI_mvc[NDVI_mvc == -1] = None
    LST_mvc = np.nanmean(LST_all, axis=2)
    VSWI = (LST_mvc/NDVI_mvc)*0.001

    # 读取一个地理参考和投影，将VSWI保存为临时文件
    dssub1 = gdal.Open(files[0])
    OutDrv = gdal.GetDriverByName('GTiff')
    OutDat = OutDrv.Create(tmp_path_file_aaa,
                           dssub1.RasterXSize, dssub1.RasterYSize, 1,
                           gdal.GDT_Float64, ["COMPRESS=DEFLATE"])
    assert OutDat is not None
    OutDat.SetGeoTransform(dssub1.GetGeoTransform())
    OutDat.SetProjection(dssub1.GetProjection())
    OutDat.GetRasterBand(1).SetNoDataValue(-2)
    OutDat.GetRasterBand(1).WriteArray(VSWI)
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

    start_year_time1 = datetime.date(int(start_time.split("-")[0]), int(start_time.split("-")[1]), int(start_time.split("-")[2]))
    start_year_time2 = datetime.date(int(end_time.split("-")[0]), int(end_time.split("-")[1]), int(end_time.split("-")[2]))

    start_files = []
    # end_files = []

    d = start_year_time1
    delta = datetime.timedelta(days=1)
    aa = ["_".join(os.path.basename(i).split("_")[0:-2]) for i in file_list]
    # aa = ["_".join(i.split("/")[-1].split("_")[0:-2]) for i in file_list]
    # print aa[1]
    while d <= start_year_time2:
        new_day = d.strftime("%Y-%m-%d")
        start_time_tmp = "".join([str(int(new_day.split("-")[0])), new_day.split("-")[1], new_day.split("-")[2]])
        # end_time_tmp = "".join([str(int(new_day.split("-")[0])+1), new_day.split("-")[1], new_day.split("-")[2]])
        start_FY3_name = "%s_VIRR_DXX_GLL_L1_%s" % (Star_num, start_time_tmp)
        # end_FY3_name = "%s_VIRR_DXX_GLL_L1_%s" % (Star_num, end_time_tmp)
        try:
            print
            # cc = aa.index(start_FY3_name) & aa.index(end_FY3_name)
            cc = aa.index(start_FY3_name)
            start_files.append(file_list[aa.index(start_FY3_name)])
            # end_files.append(file_list[aa.index(end_FY3_name)])
        except:
            print "The file %s isn't exit.please check!" % (start_FY3_name)
        d += delta

    return start_files


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
    # check files
    all_file_list = dirlist(path, [], "LDF")
    files = Check_file(all_file_list, start_time, end_time, Star_num)

    # Read, Calculate NDVI MVC, LST Mean, VSWI
    Mean_LST_MVC_NDVI(files, tmp_path)

    # Transform Project, Mask
    Mask_Transform(shp_path, out_path, tmp_path)
    # Mask(shp_path)
    # Transform(out_path)


if __name__ == '__main__':
    # FY3C_VIRR_DXX_GLL_L1_20170309_1113_1000M.LDF
    # FY3B_VIRR_DXX_GLL_L1_20160213_1516_1000M.LDF
    path = "/Users/shuailee/Documents/Climate/Example/FY3"
    shp_path = "/Users/shuailee/Documents/Climate/Example/Boundary/shaanxi.shp"
    out_path = "/Users/shuailee/Documents/Climate/tmp"
    out_name = "shaanxi_vswi_2017-1-02.tif"
    tmp_path = "/Users/shuailee/Documents/Climate/tmp"
    start_time = "2017-12-02"
    end_time = "2017-12-14"
    Star_num = "FY3B"

    if not os.path.exists(out_path):
        ccc = "mkdir %s" % out_path
        os.system(ccc)

    if not os.path.exists(tmp_path):
        ccc = "mkdir %s" % tmp_path
        os.system(ccc)

    out_path_file = os.path.join(out_path, out_name)

    main(path, shp_path, out_path_file, start_time, end_time, Star_num, tmp_path)