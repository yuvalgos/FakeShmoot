"""
A short script to extract images from videos.

Videos should be unzipped before running
"""

import cv2
import glob

###
samples_per_second = 3  # approximately
###

im_idx = 0
for file_name in glob.glob("./Data/RawVideos/*.mp4"):
    vid = cv2.VideoCapture(file_name)

    fps = vid.get(cv2.CAP_PROP_FPS)
    sample_rate = int(fps / samples_per_second)

    for i in range(int(vid.get(cv2.CAP_PROP_FRAME_COUNT))):
        success, image = vid.read()
        if i % sample_rate == 0:
            cv2.imwrite("./Data/RawImFromVid/v{}.jpg".format(im_idx), image)
            im_idx += 1

    vid.release()
