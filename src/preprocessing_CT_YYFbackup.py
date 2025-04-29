import sys
sys.path.append("..")
# from utility import *
from argparse import ArgumentParser
import os
import re
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
# import nibabel as nib
# import scipy.ndimage as ndimg
# from myvi import myvi
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import math
import datetime
import os, glob, json
import cv2
import copy
import seaborn as sns

def resample_image_to_1mm(itk_image, out_spacing=(1.0, 1.0, 1.0), is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    print(out_size)
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def resample_image_to_256x256x256(itk_image, is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [256, 256, 256]
    out_spacing = [original_spacing[0] / (out_size[0] / original_size[0]),
                   original_spacing[1] / (out_size[1] / original_size[1]),
                   original_spacing[2] / (out_size[2] / original_size[2])]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def resample_image_to_128x128x128(itk_image, is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [128, 128, 128]
    out_spacing = [original_spacing[0] / (out_size[0] / original_size[0]),
                   original_spacing[1] / (out_size[1] / original_size[1]),
                   original_spacing[2] / (out_size[2] / original_size[2])]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def resample_image_to_128x128x8(itk_image, is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [128, 128, 8]
    out_spacing = [original_spacing[0] / (out_size[0] / original_size[0]),
                   original_spacing[1] / (out_size[1] / original_size[1]),
                   original_spacing[2] / (out_size[2] / original_size[2])]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def resample_image_to_256x256x128(itk_image, is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [256, 256, 128]
    out_spacing = [original_spacing[0] / (out_size[0] / original_size[0]),
                   original_spacing[1] / (out_size[1] / original_size[1]),
                   original_spacing[2] / (out_size[2] / original_size[2])]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def resample_xy_target_size(itk_image, out_size, is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize() # x, y, z
    out_spacing = [original_spacing[0] / (out_size[2] / original_size[0]), # x
                   original_spacing[1] / (out_size[1] / original_size[1]), # y
                   original_spacing[2] / (original_size[0] / original_size[2])] # z

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize([out_size[1], out_size[2], original_size[2]]) # outsize: z, x, y, setsize: x, y, z
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def resample_xyz_target_size(itk_image, out_size, is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize() # x, y, z
    out_spacing = [original_spacing[0] / (out_size[2] / original_size[0]), # x
                   original_spacing[1] / (out_size[1] / original_size[1]), # y
                   original_spacing[2] / (out_size[0] / original_size[2])] # z

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize([out_size[1], out_size[2], out_size[0]]) # outsize: z, x, y, setsize: x, y, z
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def bbox_3D(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return [rmin, rmax, cmin, cmax, zmin, zmax]

def takedate(elem):
    return int(elem)

def bbox_3D(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return [rmin, rmax, cmin, cmax, zmin, zmax]

def image_compose(npImage_resample_adjust, x=2, z=2):

    [sz,sx,sy] = npImage_resample_adjust.shape
    combine_x = np.zeros([x * sz, z * sy])  # 创建一个新图
    combine_z = np.zeros([x * sx, z * sy])  # 创建一个新图

    total_num = 0
    for ix in range(x):
        for iz in range(z):
            print(ix,iz)
            combine_x[ix*sz:(ix+1)*sz, iz*sy:(iz+1)*sy] = npImage_resample_adjust[::-1,sx//5*(ix*x+iz+1),::]
            combine_z[ix*sx:(ix+1)*sx, iz*sy:(iz+1)*sy] = npImage_resample_adjust[sz//5*(ix*x+iz+1),:,::]

    return combine_x, combine_z

# def get_time_swab(patient_id, clinical_data_feature):
#
#     time1 = clinical_data_feature.loc[clinical_data_feature['Pseudonym'] == patient_id].values[0][
#         2+3]  # Date_of_Positive_Covid_Swab
#     time2 = clinical_data_feature.loc[clinical_data_feature['Pseudonym'] == patient_id].values[0][
#         3+3]  # Date_of_acquisition_of_1st_RT-PCR
#     time3 = clinical_data_feature.loc[clinical_data_feature['Pseudonym'] == patient_id].values[0][
#         4+3]  # Date_of_acquisition_of_1st_RT-PCR result
#     time4 = clinical_data_feature.loc[clinical_data_feature['Pseudonym'] == patient_id].values[0][
#         5+3]  # Date_of_acquisition_of_2st_RT-PCR
#     time5 = clinical_data_feature.loc[clinical_data_feature['Pseudonym'] == patient_id].values[0][
#         6+3]  # Date_of_acquisition_of_2st_RT-PCR
#
#
#     if not pd.isnull(time1):
#         date_slice_str = time1.split(' ')[0]
#         yy_slice = int(date_slice_str.split('/')[2])
#         mm_slice = int(date_slice_str.split('/')[1])
#         dd_slice = int(date_slice_str.split('/')[0])
#         data_str = date_slice_str
#         date = datetime.datetime(yy_slice, mm_slice, dd_slice)
#     elif not pd.isnull(time2):
#         date_slice_str = time2.split(' ')[0]
#         yy_slice = int(date_slice_str.split('/')[2])
#         mm_slice = int(date_slice_str.split('/')[1])
#         dd_slice = int(date_slice_str.split('/')[0])
#         data_str = date_slice_str
#         date = datetime.datetime(yy_slice, mm_slice, dd_slice)
#     elif not pd.isnull(time3):
#         date_slice_str = time3.split(' ')[0]
#         yy_slice = int(date_slice_str.split('/')[2])
#         mm_slice = int(date_slice_str.split('/')[1])
#         dd_slice = int(date_slice_str.split('/')[0])
#         data_str = date_slice_str
#         date = datetime.datetime(yy_slice, mm_slice, dd_slice)
#     elif not pd.isnull(time4):
#         date_slice_str = time4.split(' ')[0]
#         yy_slice = int(date_slice_str.split('/')[2])
#         mm_slice = int(date_slice_str.split('/')[1])
#         dd_slice = int(date_slice_str.split('/')[0])
#         data_str = date_slice_str
#         date = datetime.datetime(yy_slice, mm_slice, dd_slice)
#     elif not pd.isnull(time5):
#         date_slice_str = time5.split(' ')[0]
#         yy_slice = int(date_slice_str.split('/')[2])
#         mm_slice = int(date_slice_str.split('/')[1])
#         dd_slice = int(date_slice_str.split('/')[0])
#         data_str = date_slice_str
#         date = datetime.datetime(yy_slice, mm_slice, dd_slice)
#     else:
#         # yy_slice = int(2099)
#         # mm_slice = int(01)
#         # dd_slice = int(01)
#         data_str = ''
#         date = ''
#
#     return date, data_str

# def get_time_ct_scan(scan_name):
#     yy_slice = int(scan_name[0:4])
#     mm_slice = int(scan_name[4:6])
#     dd_slice = int(scan_name[6:8])
#     date = datetime.datetime(yy_slice, mm_slice, dd_slice)
#     return date

# def quality_check(scan_name, data_path, save_image, save_mask, invalidbody=[str("HEART")], total_slice=150, resize_target = [200, 350, 350]):
def quality_check(scan_name, condition, resize_target = [200, 350, 350], save_resample=False):

    # if scan_name == 'Covid52309_20210204_0_8.nii.gz':
    #     print("here")
    # initial
    ck_no = 'blank'
    thick_num = 9999
    rnum = -1


    img_name = scan_name
    jason_name = img_name.replace('.nii.gz','.json')
    seg_name = scan_name
    j = img_name

    # 1. check body
    img_path  = '/media/NAS02/yyfang/CAM_covid/raw_all/CT_img/' + img_name
    json_path = '/media/NAS02/yyfang/CAM_covid/raw_all/CT_jason/' + jason_name
    lung_path = '/media/NAS02/yyfang/CAM_covid/raw_all/CT_ggo_consolid/' + seg_name
    save_image= '/media/NAS02/yyfang/CAM_covid/process_all/CT_img_3D/' + img_name
    save_mask = '/media/NAS02/yyfang/CAM_covid/process_all/CT_ggo_consolid/' + img_name

    with open(json_path, 'r') as jsonfile:
        json_dict = json.load(jsonfile)

    if 'kernel' in condition:

        if str('ConvolutionKernel_1') in json_dict.keys():
            ConvolutionKernel = str(json_dict['ConvolutionKernel_1'])
        elif str('ConvolutionKernel') in json_dict.keys():
            ConvolutionKernel = str(json_dict['ConvolutionKernel'])
        else:
            ConvolutionKernel = "00" # have no kernel

        # if ConvolutionKernel == "B":
        #     blank = 0
        ck_no = "".join(list(filter(str.isdigit, ConvolutionKernel)))
        if ck_no == '': # characteristic
            # ck_no = ConvolutionKernel
            if ConvolutionKernel == 'SOFT':
                ck_no = '01'
            elif ConvolutionKernel == 'CHST':
                ck_no = '02'
            elif ConvolutionKernel == 'LUNG':
                ck_no = '03'
            elif ConvolutionKernel == 'STANDARD':
                ck_no = '04'
            elif ConvolutionKernel == 'BONEPLUS':
                ck_no = '05'
            elif ConvolutionKernel == 'B':
                ck_no = '06'
            elif ConvolutionKernel == 'C':
                ck_no = '07'
            elif ConvolutionKernel == 'L':
                ck_no = '08'
            elif ConvolutionKernel == 'YA':
                ck_no = '09'
            elif ConvolutionKernel == 'YB':
                ck_no = '10'
            else:
                ck_no = '00'
        try:
            ck_no = int(ck_no)
        except:
            blank = 0
        #     return 0, ck_no
        # elif
        #     return 0, ck_no

    if 'SliceThickness' in condition:
        # 3. thickkness
        thick_num = json_dict['SliceThickness']
        # if thick_num > 3.0:
        #     # print(j + ": invalidate thickness")
        #     return 2
                    # break

    if 'z_number' in condition:
        # check Z-axis
        img_stik = sitk.ReadImage(img_path)
        img_np = sitk.GetArrayFromImage(img_stik)
        lung_mask_sitk = sitk.ReadImage(lung_path)
        lung_mask_np = sitk.GetArrayFromImage(lung_mask_sitk)
        # Make the lung mask binary [0, 1, 2, 3] --> [0, 1]
        binary_lung_mask_np = copy.deepcopy(lung_mask_np)
        binary_lung_mask_np[binary_lung_mask_np != 0] = 1
        lung_img_np = img_np
        lung_img_np[binary_lung_mask_np == 0] = -1024
        try:
            rmin, rmax, cmin, cmax, zmin, zmax = bbox_3D(binary_lung_mask_np)
            # if rmax - rmin < 100:
            rnum = rmax - rmin
                # print(j + ": insufficient slices on Z-axis")
                # return 3
        except:
            rnum = 0
            # print(j + ": insufficient slices on Z-axis")
            # return 3



    # sum_pixel = cropped_binary_lung_mask_np.shape[1] * cropped_binary_lung_mask_np.shape[2]
    # plane_select = [t for t in range(cropped_binary_lung_mask_np.shape[0]) if
    #                 cropped_binary_lung_mask_np[t, :, :].sum() / sum_pixel > 0.3]
    # if len(plane_select) < 20: # 应该换高一点
    #     print(j + ": invalidate slice number")
    #     return 1

    if save_resample and thick_num <= 3.0 and rnum>=100:
        # Crop out Lung
        # cropped_binary_lung_mask_np = binary_lung_mask_np[rmin:rmax, cmin:cmax, zmin:zmax]
        if os.path.exists(save_image):
            print("exist!")
            return ck_no, thick_num, rnum
        cropped_lung_region_np = lung_img_np[rmin:rmax, cmin:cmax, zmin:zmax]
        cropped_lung_mask_np   = lung_mask_np[rmin:rmax, cmin:cmax, zmin:zmax]
        # resampled
        cropped_lung_region_sitk = sitk.GetImageFromArray(cropped_lung_region_np)
        cropped_lung_region_sitk.SetSpacing(img_stik.GetSpacing()) # getsize: x, y, z resize_target: z, x, y
        cropped_lung_mask_sitk = sitk.GetImageFromArray(cropped_lung_mask_np)
        cropped_lung_mask_sitk.SetSpacing(lung_mask_sitk.GetSpacing())

        resampled_cropped_lung_region_sitk = resample_xyz_target_size(cropped_lung_region_sitk, resize_target)
        resampled_cropped_lung_mask_sitk = resample_xyz_target_size(cropped_lung_mask_sitk, resize_target, is_label=True)

        sitk.WriteImage(resampled_cropped_lung_region_sitk, save_image)
        sitk.WriteImage(resampled_cropped_lung_mask_sitk, save_mask)
        # print(j + ": success")

    return ck_no, thick_num, rnum


def image_3D_normalisation(npImage, min_value=-1024, max_value=-100):

    # crop
    npImage_norm = npImage
    npImage_norm[npImage < min_value] = min_value
    npImage_norm[npImage > max_value] = max_value

    # norm
    npImage_norm = (npImage_norm-min_value)/(max_value-min_value)

    # normalization: x-y
    # npImage_resample_adjust1 = (npImage_resample_adjust - min_value) / (max_value - min_value)
    # slice = npImage_resample_adjust1[16, :, :]
    # print("调节窗口窗位之后CT值的范围位为{}~{}".format(np.min(slice), np.max(slice)))
    # plt.figure(figsize=(5, 5))
    # plt.imshow(slice, 'gray')
    # plt.show()

    return npImage_norm