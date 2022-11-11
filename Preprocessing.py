import os
import cv2
import numpy as np
from tqdm import tqdm

def Preprocessing(in_path, targets_out_path, inputs_out_path):

    img_names = os.listdir(in_path)
    img_paths = [in_path + n for n in img_names]
    listed = zip(img_paths, img_names)

    for j, i in tqdm(listed, total=len(img_paths), desc='ji-loop'):

        img = cv2.imread(j, 0)

        inv = np.bitwise_not(img)

        height=img.shape[0]
        width =img.shape[1]

        if height > width:
            h=0
            w=(height-width)//2 #Padding both sides to make a square.

        elif height < width:
            h=(width-height)//2 #Padding top and bottom to make a square.
            w=0

        else:
            h=0
            w=0

        padded = cv2.copyMakeBorder(src=inv, top=h, bottom=h, left=w, right=w, borderType=cv2.BORDER_CONSTANT, value=0)

        resized = cv2.resize(padded,(512,512))

        thv, denv = cv2.threshold(resized, 25, 255, cv2.THRESH_TOZERO)

        cv2.imwrite(targets_out_path + i, denv)

        blurred = cv2.GaussianBlur(denv,(3,3), sigmaX=0, sigmaY=0)

        edged = cv2.Canny(image=blurred, threshold1=100, threshold2=200) 

        cv2.imwrite(inputs_out_path + i, edged)

    print('Done preprocessing.')

Data = Preprocessing(in_path='/path/to/the/raw/images/directory/',
                     targets_out_path='/path/to/the/target/images/directory/',
                     inputs_out_path='/path/to/the/input/images/directory/')
