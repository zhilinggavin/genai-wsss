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