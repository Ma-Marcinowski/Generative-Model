import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.fftpack import dctn

from tqdm import tqdm

def DCT(in_path, out_path, out_name):

    img_names = os.listdir(in_path)
    img_paths = [in_path + n for n in img_names]
    listed = zip(img_paths, img_names)     

    averages = []        

    for j, i in tqdm(listed, total=len(img_paths), desc='ji-loop'):

        img = cv2.imread(j, 0)

        resized = cv2.resize(img,(512,512)) #To ensure that results from the generator have the same size as real images.

        transformed = dctn(resized, norm='ortho') #Two dimensional DCT.

        logged = np.log(np.abs(transformed)) #Logarithmic scale, to ensure that higher frequencies are visible (note that log takes only positive values, hence abs).

        averages.append(logged)

    avraged = np.mean(averages, 0)

    hmap = sns.heatmap(data=avraged, square=True, xticklabels=False, yticklabels=False)

    plt.savefig(out_path + out_name + '.png')
    plt.clf() #Clears the plot, to ensure that it will not affect the next one.

Trues = DCT(in_path='/path/to/the/true/images/directory/',
            out_path='/path/to/the/dct/images/directory',
            out_name='name_of_the_dct_image')

Fakes = DCT(in_path='/path/to/the/fake/images/directory/',
            out_path='/path/to/the/dct/images/directory',
            out_name='name_of_the_dct_image')
