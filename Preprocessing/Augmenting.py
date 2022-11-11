import os
import cv2
import numpy as np
from tqdm import tqdm

import elasticdeform as edf

def Augmenting(in_path, out_path):

    img_names = os.listdir(in_path)
    img_paths = [in_path + n for n in img_names]
    listed = zip(img_paths, img_names)

    for j, i in tqdm(listed, total=len(img_paths), desc='ji-loop'):

        img = cv2.imread(j, 0)

        img = edf.deform_grid(img,
                              displacement=np.random.uniform(0, 50, (2, 5, 5)),
                              mode='constant',
                              cval=0.0,
                              prefilter=None,
                              axis=(0, 1))

        r = np.random.randint(0, 256, 2)
        img[r[0]:r[0]+128,r[1]:r[1]+128] = 0.0

        cv2.imwrite(out_path + i, img)

    print('Done augmenting.')

Data = Augmenting(in_path='/path/to/the/input/images/directory/',
                  out_path='/path/to/the/augmented/images/directory')
