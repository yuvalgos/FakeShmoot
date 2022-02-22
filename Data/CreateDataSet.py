"""
script to creat data set of small squared images from the raw
images and video raw images (already extracted from the video)

Images should be unzipped before running
"""


import cv2
import glob
from pathlib import Path


###
im_size = 256
dir = "Data/DataSet" + str(im_size) + "/"
###


def create_data_set(source_dir, dest_dir):
    idx = 0
    for file_name in glob.glob(source_dir + "*"):
        img = cv2.imread(file_name)

        # cut to square
        h, w = img.shape[:2]
        if h > w:
            img = img[int(h/2-w/2): int(h/2+w/2), :, :]
        else:
            img = img[:, int(w/2-h/2): int(w/2+h/2), :]

        # resize
        img = cv2.resize(img, (im_size, im_size), interpolation=cv2.INTER_AREA)

        # save
        cv2.imwrite(dest_dir + str(idx) + ".jpg", img)
        idx += 1


Path(dir + "FromIm").mkdir(parents=True, exist_ok=True)
create_data_set("Data/RawImages/", dir + "FromIm/")

Path(dir + "FromVid").mkdir(parents=True, exist_ok=True)
create_data_set("Data/RawImFromVid/", dir + "FromVid/")
