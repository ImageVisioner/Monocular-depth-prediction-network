'''
中文：此段代码用来提取mat文件的图像数据。用来转换为png格式
并且分别提取放在三个文件夹：nyu_depths / nyu_images / nyu _labels
注意：请更改路径（h5py.file）

Eng：
This code is used to extract the image data from mat files.
 It is used to convert to png format
and extracted in three folders respectively: nyu_depths / nyu_images / nyu _labels
Note: Please change the path (h5py.file)
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
import os
from PIL import Image
import cv2

f = h5py.File("nyu_depth_v2_labeled.mat")

# extract images
images = f["images"]
images = np.array(images)

path_converted = '.\\NYU\\nyu_images'
if not os.path.isdir(path_converted):
    os.makedirs(path_converted)

images_number = []
for i in range(len(images)):
    images_number.append(images[i])
    a = np.array(images_number[i])
    r = Image.fromarray(a[0]).convert('L')
    g = Image.fromarray(a[1]).convert('L')
    b = Image.fromarray(a[2]).convert('L')
    img = Image.merge("RGB", (r, g, b))
    img = img.transpose(Image.ROTATE_270)
    iconpath = '.\\NYU\\nyu_images/' + str(i) + '.png'
    img.save(iconpath, optimize=True)
    # exit(0)

print("image extract finished!!!!!")

# extract depths
depths = f["depths"]
depths = np.array(depths)

path_converted = '.\\NYU\\nyu_depths/'
if not os.path.isdir(path_converted):
    os.makedirs(path_converted)

max = depths.max()
depths = depths / max * 65535
depths = depths.transpose((0, 2, 1))

for i in range(len(depths)):
    print(str(i) + '.png')
    depths_img = np.uint16(depths[i])
    depths_img_new = cv2.flip(depths_img, 1)
    print(depths_img_new.max(), depths_img_new.min())
    iconpath = path_converted + str(i) + '.png'
    cv2.imwrite(iconpath, depths_img_new)
    # exit(0)
print("depths extract finished!!!!!")


labels = f["labels"]
labels = np.array(labels)

path_converted = '.\\NYU\\nyu_labels/'
if not os.path.isdir(path_converted):
    os.makedirs(path_converted)

labels_max = labels.max()
labels = labels / labels_max * 65535
labels = labels.transpose((0, 2, 1))

for i in range(len(labels)):

    label_img = np.uint16(labels[i])
    label_img_new = cv2.flip(label_img, 1)

    iconpath = '.\\NYU\\nyu_labels/' + str(i) + '.png'
    cv2.imwrite(iconpath, label_img_new)
    # exit(0)

print("labels extract finished!!!!!")


