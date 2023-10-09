import cv2
import numpy as np

a = np.load('/home/ana/Study/CVPR/lego/dataset/20200820-subject-03/20200820_135508/836212060125/labels_000001.npz')
b = np.load('/home/ana/Study/CVPR/lego/dataset/20200820-subject-03/20200820_135508/pose.npz')
for i in a:
    print(i)



print(a['joint_2d'])
print(a['joint_3d'])