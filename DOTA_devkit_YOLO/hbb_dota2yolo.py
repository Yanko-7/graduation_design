import os
from YOLO_Transform import *
import dota_utils as utils
import ImgSplit_multi_process as Split
def gen_yolo_hbb():
    dota2Darknet(imgpath='',
                txtpath= '',
                dstpath='',
                extractclassname='')
    dota2Darknet(imgpath='',
                txtpath= '',
                dstpath='',
                extractclassname='')

def split_hbb_dota():
    split = Split.splitbase(r'example',
                    r'example_split',
                    gap=200,        # 
                    subsize=1024,   # 
                    num_process=8
                    )
    split.splitdata(1)  

if __name__ == '__main__':
    split_hbb_dota()
    gen_yolo_hbb()