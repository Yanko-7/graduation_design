import os
from YOLO_Transform import *
import dota_utils as utils
import ImgSplit_multi_process as Split

# input 'Path/to/trainPath' / 'path/to/valPath'
# trainPath/
#       -images/Pxxx.png
#       -labelTxt/pXXX.txt
# output 'path/to/example_split'
trainPath = r'/Users/Yanko/Documents/graduation_design/utils/DOTA_devkit_YOLO/example_train'
valPath = r'/Users/Yanko/Documents/graduation_design/utils/DOTA_devkit_YOLO/example_val'

trainOutPath = trainPath + '_split'
valOutPath = valPath + '_split'

FinalOutPath = r'/Users/Yanko/Documents/graduation_design/utils/DOTA_devkit_YOLO/xxxx'
def gen_yolo_hbb():
    dota2Darknet(imgpath=os.path.join(trainOutPath,'images'),
                txtpath=os.path.join(trainOutPath,'labelTxt'),
                dstpath=os.path.join(trainOutPath,'yolo_labelTxt'),
                extractclassname=utils.wordname_15)
    dota2Darknet(imgpath=os.path.join(valOutPath,'images'),
                txtpath=os.path.join(valOutPath,'labelTxt'),
                dstpath=os.path.join(valOutPath,'yolo_labelTxt'),
                extractclassname=utils.wordname_15)

def split_hbb_dota():
    split1 = Split.splitbase(trainPath,
                    trainOutPath,
                    gap=200,        # 
                    subsize=1024,   # 
                    num_process=8
                    )
    split1.splitdata(1)
    split2 = Split.splitbase(valPath,
                    valOutPath,
                    gap=200,        # 
                    subsize=1024,   # 
                    num_process=8
                    )
    split2.splitdata(1)

def move_files_from_A_to_B(A, B):
    # 确保目标文件夹存在
    if not os.path.exists(B):
        os.makedirs(B)

    # 获取A文件夹中的所有文件
    files = os.listdir(A)

    # 遍历所有文件，并将它们移动到B文件夹
    for file in files:
        src = os.path.join(A, file)
        dst = os.path.join(B, file)
        shutil.move(src, dst)

if __name__ == '__main__':
    split_hbb_dota()
    gen_yolo_hbb()
    if os.path.exists(FinalOutPath):
        shutil.rmtree(FinalOutPath)  # delete output folder
    os.makedirs(FinalOutPath)  # make new output folder
    os.makedirs(os.path.join(FinalOutPath,'images'))
    img_train = os.path.join(os.path.join(FinalOutPath,'images'),'train')
    img_val = os.path.join(os.path.join(FinalOutPath,'images'),'val')
    os.makedirs(img_train)
    os.makedirs(img_val)
    os.makedirs(os.path.join(FinalOutPath,'labelTxt'))
    label_train = os.path.join(os.path.join(FinalOutPath,'labelTxt'),'train')
    label_val = os.path.join(os.path.join(FinalOutPath,'labelTxt'),'val')
    os.makedirs(label_train)
    os.makedirs(label_val)

    move_files_from_A_to_B(os.path.join(trainOutPath,'images'), img_train)
    move_files_from_A_to_B(os.path.join(trainOutPath,'yolo_labelTxt'), label_train)
    
    move_files_from_A_to_B(os.path.join(valOutPath,'images'), img_val)
    move_files_from_A_to_B(os.path.join(valOutPath,'yolo_labelTxt'), label_val)
    print(f'output in {FinalOutPath}')