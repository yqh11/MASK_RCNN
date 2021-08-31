# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:59:06 2021

@author: louyu
"""

import os
import shutil
import time
import sys
import importlib
importlib.reload(sys)

def copy_and_rename(fpath_input, fpath_output):
    for file in os.listdir(fpath_input):
        for inner in os.listdir(fpath_input + file + '/'):
            print(inner)
            if os.path.splitext(inner)[0] == 'label':
                former = os.path.join(fpath_input, file)
                oldname = os.path.join(former, inner)
                print(oldname)
                newname_1 = os.path.join(fpath_output, file.split('_')[0] + '.png')
                #os.rename(oldname, newname)
                shutil.copyfile(oldname, newname_1)
                
                
if __name__ == '__main__':
    print('start')
    
    fpath_input = './temp/labelme_json/'
    fpath_output = './temp/cv2_mask/'
    copy_and_rename(fpath_input, fpath_output)
    
print('end')
    