import os
import json
import math
from collections import defaultdict
import numpy
import glob
from PIL import Image
import sys
import shapely.geometry as shgeo
labels = ['plane','ship','storage-tank','baseball-diamond','tennis-court','basketball-court',
          'ground-track-field','harbor','bridge','large-vehicle','small-vehicle', 'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool']
#生成训练集标签，格式
# 图片路径 框1 框2
# 框的格式 id,x1,y1,x2,y2,a
# 数据集解压到同级文件夹
#生成trainlist.txt,testlist.txt
def parse_dota_poly(filename):
    """
        parse the dota ground truth in the format:
        [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    objects = []
    #print('filename:', filename)
    f = []
    if (sys.version_info >= (3, 5)):
        fd = open(filename, 'r')
        f = fd
    elif (sys.version_info >= 2.7):
        fd = codecs.open(filename, 'r')
        f = fd
    # count = 0
    while True:
        line = f.readline()
        # count = count + 1
        # if count < 2:
        #     continue
        if line:
            splitlines = line.strip().split(' ')
            object_struct = {}
            ### clear the wrong name after check all the data
            #if (len(splitlines) >= 9) and (splitlines[8] in classname):
            if (len(splitlines) < 9):
                continue
            if (len(splitlines) >= 9):
                    object_struct['name'] = splitlines[8]
            if (len(splitlines) == 9):
                object_struct['difficult'] = '0'
            elif (len(splitlines) >= 10):
                # if splitlines[9] == '1':
                # if (splitlines[9] == 'tr'):
                #     object_struct['difficult'] = '1'
                # else:
                object_struct['difficult'] = splitlines[9]
                # else:
                #     object_struct['difficult'] = 0
            object_struct['poly'] = [(float(splitlines[0]), float(splitlines[1])),
                                     (float(splitlines[2]), float(splitlines[3])),
                                     (float(splitlines[4]), float(splitlines[5])),
                                     (float(splitlines[6]), float(splitlines[7]))
                                     ]
            gtpoly = shgeo.Polygon(object_struct['poly'])
            object_struct['area'] = gtpoly.area
            # poly = list(map(lambda x:np.array(x), object_struct['poly']))
            # object_struct['long-axis'] = max(distance(poly[0], poly[1]), distance(poly[1], poly[2]))
            # object_struct['short-axis'] = min(distance(poly[0], poly[1]), distance(poly[1], poly[2]))
            # if (object_struct['long-axis'] < 15):
            #     object_struct['difficult'] = '1'
            #     global small_count
            #     small_count = small_count + 1
            objects.append(object_struct)
        else:
            break
    return objects
# from ../utils/DOTA_devkit_YOLO


def generate_dataset(path, save_name, txtpath ,patch = 1,train = True):
    img_name = list(filter(lambda x:x[-3:]=='png',os.listdir(path)))
    total_line = ""
    if train:
        total_line+=f"patch:{patch}\n"
    cur_labels = set()
    for iname in img_name:
        ip = path + '/'+iname
        ip_label = (txtpath + '/' + iname).split('.')[0] + '.txt'
        # ip_json = ip.split('.')[0]+'.json'
        # data =json.load(open(ip_json,'r',encoding='utf-8'))
        root_path = "dataset/"+ip
        with Image.open(ip) as img:
            ih, iw = img.size
        # iw = data["imageHeight"]
        # ih = data["imageWidth"]
        meshxy = defaultdict(list)
        boxes = []
        objects = parse_dota_poly(ip_label)
        for obj in objects:
            label = labels.index(obj['name'])
            cur_labels.add(label)
            points = obj['poly']
            w = ((points[0][0]-points[1][0])**2+(points[0][1]-points[1][1])**2)**0.5
            h = ((points[1][0]-points[2][0])**2+(points[1][1]-points[2][1])**2)**0.5
            if w>h:
                if points[1][0] == points[0][0]:
                    a = math.pi / 2.0
                else:
                    a = math.atan((points[1][1] - points[0][1]) / (points[1][0] - points[0][0]))
            else:
                if points[1][0]  == points[2][0]:
                    a = math.pi / 2.0
                else:
                    a = math.atan((points[2][1] - points[1][1]) / (points[2][0] - points[1][0]))
                w,h = h,w
            cx = (points[0][0]+points[2][0])/2.0
            cy = (points[0][1]+points[2][1])/2.0
            a = round(a, 4)
            keypoint = [max(int(cx - w/2.0),0),max(int(cy - h/2.0),0),min(int(cx+w/2.0),iw),min(int(cy+h/2.0),ih)]
            print(keypoint)
            if patch>1:
                sw = iw//patch
                sh = ih//patch
                s1 = keypoint[0]//sw+1
                e1 = math.ceil(keypoint[2]/sw)
                meshx = [keypoint[0]]+[i*sw for i in range(s1,e1)]+[keypoint[2]]
                s2 = keypoint[1]//sh+1
                e2 = math.ceil(keypoint[3]/sh)
                meshy = [keypoint[1]]+[i*sh for i in range(s2,e2)]+[keypoint[3]]
                for i in range(len(meshx)-1):
                    for j in range(len(meshy)-1):
                        num_grid = (meshy[j]//sh)*patch+(meshx[i]//sw)
                        line = str(label)+","+str(meshx[i]%sw)+","+str(meshy[j]%sh)+","+\
                               str((meshx[i+1]-1)%sw)+","+str((meshy[j+1]-1)%sh)+","+str(a)
                        meshxy[num_grid].append(line)
            else:
                line = str(label) + "," + ",".join(list(map(str,keypoint))) + "," + str(a)
                boxes.append(line)
        if patch>1:
            for key,value in meshxy.items():
                lines = root_path+" "+str(key)+" "+" ".join(value)+"\n"
                total_line+=lines
        else:
            lines = root_path+" "+" ".join(boxes)+"\n"
            total_line+=lines
    with open(save_name,"w") as f:
        f.write(total_line)
    print(sorted(list(cur_labels)))
if __name__=="__main__":
    generate_dataset('train','trainlist.txt','train' ,patch=4,train=True)
    generate_dataset('val', 'vallist.txt', 'val',patch=1,train=False)
