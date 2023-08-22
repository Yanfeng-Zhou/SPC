import numpy as np
import argparse
import os
import SimpleITK as sitk
from mayavi import mlab

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_path', default='//10.0.5.233/shared_data/SPC/seg_pred/test/LA/best_result2_Jc_0.9013')
    # parser.add_argument('--seg_path', default='//10.0.5.233/shared_data/XNet/dataset/Atrial/val/mask')
    args = parser.parse_args()

    for image in os.listdir(args.seg_path):
        seg_path = os.path.join(args.seg_path, image)
        image_name = os.path.splitext(image)[0]
        print(image_name)

        image = sitk.ReadImage(seg_path)
        image = sitk.GetArrayFromImage(image)

        figure = mlab.figure(image_name, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
        mlab.contour3d(image, contours=[1], color=(1, 0, 0), opacity=0.5, figure=figure)

    mlab.show()

