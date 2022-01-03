import os, sys, random
import xml.etree.ElementTree as ET
from glob import glob
import pandas as pd
from shutil import copyfile

annotations = glob('Annotations_train/*.xml')

df = []
cnt = 0
for file in annotations:
    filename = file.split('\\')[-1]
    filename = filename.split('.')[0]
    filename = filename.split('/')[1] + '.jpg'
    row = []
    parsedXML = ET.parse(file)
    for node in parsedXML.getroot().iter('object'):
        blood_cells = node.find('name').text
        xmin = int(node.find('bndbox/xmin').text)
        xmax = int(node.find('bndbox/xmax').text)
        ymin = int(node.find('bndbox/ymin').text)
        ymax = int(node.find('bndbox/ymax').text)

        row = [filename, blood_cells, xmin, xmax, ymin, ymax]
        df.append(row)
        cnt += 1

data = pd.DataFrame(df, columns=['image_name', 'cell_type', 'xmin', 'xmax', 'ymin', 'ymax'])

data[['image_name', 'cell_type', 'xmin', 'xmax', 'ymin', 'ymax']].to_csv('train.csv', index=False)
 