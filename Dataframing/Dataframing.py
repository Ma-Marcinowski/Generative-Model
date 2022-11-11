import os
import re
import csv
import numpy as np
from tqdm import tqdm

def Dataframe(averages_path, inputs_path, targets_path, 
              averages_path_in_df, inputs_path_in_df, targets_path_in_df, 
              df_path):

    avrerages = os.listdir(averages_path)
    inputs = os.listdir(inputs_path)
    targets = os.listdir(targets_path)

    with open(df_path, 'a+') as f:

        writer = csv.writer(f)
        writer.writerow(['MaskImages', 'InputImages', 'TargetImages'])

        for i in tqdm(avrerages, desc='i-loop', leave=False):
            for j in tqdm(inputs, desc='j-loop', leave=False):
                for k in tqdm(targets, desc='k-loop', leave=False):

                    if i[:11] == j[:11] == k[:11] and j == k:
                                                
                        df_data = [averages_path_in_df + i, inputs_path_in_df + j, targets_path_in_df + k]

                        writer.writerow(df_data)
                                                        
    print('Done dataframing.')

Dataframe = Dataframe(averages_path='/path/to/the/averaged/images/directory/', 
                      inputs_path='/path/to/the/input/images/directory/',
                      targets_path='/path/to/the/target/images/directory/', 
                      averages_path_in_df='/path/to/the/averaged/images/saved/in/the/dataframe/', 
                      inputs_path_in_df='/path/to/the/input/images/saved/in/the/dataframe/', 
                      targets_path_in_df='/path/to/the/target/images/saved/in/the/dataframe/',
                      df_path='/path/to/the/dataframe.csv')
