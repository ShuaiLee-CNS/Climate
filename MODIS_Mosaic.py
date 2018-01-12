# -*- encoding: utf-8 -*-

import sys
import os
import gdal
import numpy as np
import natsort
# %matplotlib inline


def Get_Layer(ss, name):
    aa = []
    for sd, des in ss.GetSubDatasets():
        # print sd
        if sd.endswith(name):
            aa = sd
    return aa


def mk_dir_file(file_name, out_path):

    year1 = file_name.split(".")[1][1:5]
    out_path1 = os.path.join(out_path, year1)
    if os.path.exists(out_path1):
        print ("%s is exist!" % out_path1)
    else:
        ccc = "mkdir %s" % out_path1
        os.system(ccc)

    file_name = file_name + ".tif"
    out_path_file_name = os.path.join(out_path1, file_name)
    return out_path_file_name


def Mosaic(input_path1, input_path2, name, file_name, out_path):
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

    out_path_file_name = mk_dir_file(file_name, out_path)
    # 读取一个地理参考和投影，将last_data保存为临时文件
    layer_name1 = Get_Layer(ss1, name)
    dssub1 = gdal.Open(layer_name1)
    OutDrv = gdal.GetDriverByName('GTiff')
    OutDat = OutDrv.Create(out_path_file_name,
                           dssub1.RasterXSize * 2, dssub1.RasterYSize, 1,
                           gdal.GDT_Float64, ["COMPRESS=DEFLATE"])
    assert OutDat is not None
    OutDat.SetGeoTransform(dssub1.GetGeoTransform())
    aa = dssub1.GetProjection()
    OutDat.SetProjection(dssub1.GetProjection())
    OutDat.GetRasterBand(1).WriteArray(data4)
    OutDat = None


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


def Check_Take(file_list, out_files, out_path):
    #取出此时间段的所有hdf数据集
    #MOD13Q1.A2016001.h26v05.006.2016029065706
    name = "250m 16 days NDVI"

    in_aa = [".".join(os.path.basename(i).split(".")[0:3]) for i in file_list]
    out_aa = [".".join(os.path.basename(i).split(".")[0:2]) for i in out_files]
    in_aa_1 = list(set([".".join(os.path.basename(i).split(".")[0:2]) for i in file_list]))
    in_aa_1 = natsort.natsorted(in_aa_1)

    for i in range(len(in_aa_1)):
        in_tmp = in_aa_1[i]
        try:
            cc = out_aa.index(in_tmp)
            print ("%s.tif is exist!" % in_tmp)
        except:
            h_26 = "%s.h26v05" % in_tmp
            h_27 = "%s.h27v05" % in_tmp
            try:
                dd = (in_aa.index(h_26)) & (in_aa.index(h_27))
                h_26_file = file_list[in_aa.index(h_26)]
                h_27_file = file_list[in_aa.index(h_27)]
                Mosaic(h_26_file, h_27_file, name, in_tmp, out_path)
                print("%s.tif is create, congratulations!" % in_tmp)
            except:
                print ("%s %s is not exist! please check!" % (h_26, h_27))


def main(path, out_path):
    #查找未拼接的数据
    out_files = dirlist(out_path, [], "tif")
    file_list = dirlist(path, [], "hdf")

    Check_Take(file_list, out_files, out_path)



if __name__ == '__main__':

    path = "/Users/shuailee/Documents/Climate/Example/MOD13Q1"
    out_path = "/Users/shuailee/Documents/Climate/test"

    main(path, out_path)