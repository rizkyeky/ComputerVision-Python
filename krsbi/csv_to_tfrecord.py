"""
run:
python .\generate_tfrecord.py --csv_input .\test_labels.csv --image_dir .\test\ --labelmap_dir .\label_map.pbtxt --output_path test.record
"""
# python csv_to_tfrecord.py --csv_input train_labels.csv --image_dir /train --labelmap_dir label_map.pbtxt --output_path train.record

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd


import tensorflow as tf

from PIL import Image, ImageFile
# from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--csv_input', help='test, val, or train', required=True)
parser.add_argument('--output_path', help='test, val, or train', required=True)
parser.add_argument('--image_dir', help='test, val, or train', required=True)
parser.add_argument('--labelmap_dir', help='test, val, or train', required=True)
args = parser.parse_args()

ImageFile.LOAD_TRUNCATED_IMAGES = True

# flags = tf.app.flags
# flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
# flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
# flags.DEFINE_string('image_dir', '', 'Path to images')
# flags.DEFINE_string('labelmap_dir', '', 'Path to images')
# FLAGS = args

def class_text_to_int(row_label):
    name=""
    with open(args.labelmap_dir, "r") as f:
        data = f.readlines()
        for d in data:
            if d[:2] == "id":
                idx = int(d.split(": ")[-1])
            if d[:2] == "na":
                name = d.split(": ")[-1]
                name = name[1:-2] #lastchar is \n
            if name==row_label:
                return idx
    print("MISSING FROM LABELMAP:", row_label)
    return None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=height)),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=width)),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=filename)),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=filename)),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded_jpg)),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=image_format)),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),

    }))
    return tf_example

    # tf_example = tf.train.Example(features=tf.train.Features(feature={
    #     'Time': tf.train.Feature(bytes_list=tf.train.BytesList(value=[features[1].encode('utf-8')])),
    #     'Height':tf.train.Feature(int64_list=tf.train.Int64List(value=[features[2]])),
    #     'Width':tf.train.Feature(int64_list=tf.train.Int64List(value=[features[3]])),
    #     'Mean':tf.train.Feature(float_list=tf.train.FloatList(value=[features[4]])),
    #     'Std':tf.train.Feature(float_list=tf.train.FloatList(value=[features[5]])),
    #     'Variance':tf.train.Feature(float_list=tf.train.FloatList(value=[features[6]])),
    #     'Non-homogeneity':tf.train.Feature(float_list=tf.train.FloatList(value=[features[7]])),
    #     'PixelCount':tf.train.Feature(int64_list=tf.train.Int64List(value=[features[8]])),
    #     'contourCount':tf.train.Feature(int64_list=tf.train.Int64List(value=[features[9]])),
    #     'Class':tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode('utf-8')])),
    # }))
    # return tf_example

def main():
    print(args)
    writer = tf.io.TFRecordWriter(args.output_path)
    path = os.path.join(os.getcwd(), args.image_dir)
    print(path)
    examples = pd.read_csv(args.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        try:
            tf_example = create_tf_example(group, path)
            writer.write(tf_example.SerializeToString())
        except:
            print(group.filename)
            pass

    writer.close()
    output_path = os.path.join(os.getcwd(), args.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    main()