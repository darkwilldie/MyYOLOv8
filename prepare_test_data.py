import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import random
from shutil import copyfile

type45="i2,i4,i5,il100,il60,il80,io,ip,p10,p11,p12,p19,p23,p26,p27,p3,p5,p6,pg,ph4,ph4.5,ph5,pl100,pl120,pl20,pl30,pl40,pl5,pl50,pl60,pl70,pl80,pm20,pm30,pm55,pn,pne,po,pr40,w13,w32,w55,w57,w59,wo"
type45 = type45.split(',')
classes = type45

def clear_hidden_files(path):
    dir_list = os.listdir(path)
    for i in dir_list:
        abspath = os.path.join(os.path.abspath(path), i)
        if os.path.isfile(abspath):
            if i.startswith("._"):
                os.remove(abspath)
        else:
            clear_hidden_files(abspath)

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_id, annotation_dir, yolo_labels_dir):
    in_file = open(os.path.join(annotation_dir, '%s.xml' % image_id))
    out_file = open(os.path.join(yolo_labels_dir, '%s.txt' % image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    in_file.close()
    out_file.close()

wd = os.getcwd()
data_base_dir = os.path.join(wd, "VOCdevkit/")
if not os.path.isdir(data_base_dir):
    os.mkdir(data_base_dir)
work_space_dir = os.path.join(data_base_dir, "VOC2007/")
if not os.path.isdir(work_space_dir):
    os.mkdir(work_space_dir)
annotation_dir = os.path.join(work_space_dir, "Annotations/")
if not os.path.isdir(annotation_dir):
        os.mkdir(annotation_dir)
clear_hidden_files(annotation_dir)
image_dir = os.path.join(work_space_dir, "JPEGImages/")
if not os.path.isdir(image_dir):
        os.mkdir(image_dir)
clear_hidden_files(image_dir)
yolo_labels_dir = os.path.join(work_space_dir, "YOLOLabels/")
if not os.path.isdir(yolo_labels_dir):
        os.mkdir(yolo_labels_dir)
clear_hidden_files(yolo_labels_dir)
yolov8_images_dir = os.path.join(data_base_dir, "images/")
if not os.path.isdir(yolov8_images_dir):
        os.mkdir(yolov8_images_dir)
clear_hidden_files(yolov8_images_dir)
yolov8_labels_dir = os.path.join(data_base_dir, "labels/")
if not os.path.isdir(yolov8_labels_dir):
        os.mkdir(yolov8_labels_dir)
clear_hidden_files(yolov8_labels_dir)
yolov8_images_test_dir = os.path.join(yolov8_images_dir, "test/")
if not os.path.isdir(yolov8_images_test_dir):
        os.mkdir(yolov8_images_test_dir)
clear_hidden_files(yolov8_images_test_dir)
yolov8_labels_test_dir = os.path.join(yolov8_labels_dir, "test/")
if not os.path.isdir(yolov8_labels_test_dir):
        os.mkdir(yolov8_labels_test_dir)
clear_hidden_files(yolov8_labels_test_dir)

# 适配 testset 结构
testset_dir = os.path.join(wd, "集合")
clear_hidden_files(testset_dir)

test_file = open(os.path.join(wd, "yolov8_test.txt"), 'w')

for fname in os.listdir(testset_dir):
    if fname.endswith('.png') or fname.endswith('.jpg'):
        image_path = os.path.join(testset_dir, fname)
        nameWithoutExtention, _ = os.path.splitext(fname)
        annotation_path = os.path.join(testset_dir, nameWithoutExtention + '.xml')
        if os.path.exists(annotation_path):
            # 转换标签
            convert_annotation(nameWithoutExtention, testset_dir, testset_dir)
            # 复制图片和标签
            dst_img = os.path.join(yolov8_images_test_dir, fname)
            dst_lbl = os.path.join(yolov8_labels_test_dir, nameWithoutExtention + '.txt')
            copyfile(image_path, dst_img)
            copyfile(os.path.join(testset_dir, nameWithoutExtention + '.txt'), dst_lbl)
            test_file.write(dst_img + '\n')

test_file.close()
