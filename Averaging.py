import os
import re
import cv2
import numpy as np
from tqdm import tqdm

def Averaging(img_in_path, avr_out_path, lower_id_range, upper_id_ragne):

    img_names = os.listdir(img_in_path)
    img_paths = [img_in_path + n for n in img_names]

    for n in tqdm(range(lower_id_range, upper_id_ragne), desc='n-loop'):

        #Zip is an iterator, which could be travested only once (i.e. the first execution of the loop would exhaust it).
        listed = zip(img_paths, img_names) 

        averaged = []

        for j, i in tqdm(listed, total=len(img_paths), desc='ji-loop', leave=False):

            ni = re.search('[0-9]+', i).group(0)

            if int(ni) == n:

                img = cv2.imread(j, 0)

                averaged.append(img)

        avr = np.mean(averaged, 0)

        cv2.imwrite(avr_out_path + 'original_' + str(n) + '_avr.png', avr)

Train = Averaging(img_in_path='/path/to/the/target/images/directory/',
                  avr_out_path='/path/to/the/averaged/images/directory/',
                  lower_id_range=1,  #ID of the first writer to have his signature averaged.
                  upper_id_ragne=11) #ID of the last writer to have his signature averaged.
