import os
import sys
from indoor3d_util import DATA_PATH, collect_point_label

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

anno_paths = [line.rstrip() for line in open(
    os.path.join(BASE_DIR, 'meta\\anno_paths.txt'))]
anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths]

output_folder = os.path.join(ROOT_DIR, 'data\\pointclouds\\s3dis')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
for anno_path in anno_paths:
    print(anno_path)
    try:
        anno_path = anno_path.replace('\\', '/')
        elements = anno_path.split('/')
        # print(elements)
        out_filename = elements[-3]+'_' + \
            elements[-2]+'.txt'  # Area_1_hallway_1.npy
        # print(anno_path)
        # print(output_folder)
        # print(out_filename)
        collect_point_label(anno_path=anno_path, out_filename=os.path.join(
            output_folder, out_filename))
    except:
        # print(anno_path, 'ERROR!!')
        print('Error!')
