import argparse
import tensorflow as tf
import cv2
import pandas as pd
import hashlib
import sys
from pathlib import Path
import pdb


# ----- CONFIGS ----- #
# CATEGORY = {'Pedestrian':1, 'Car':2, 'Truck':3, 'Stopsign':4, 'Stopsign':5}
CATEGORY = {'Stopsign':1}

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def make_tfexample(img_array, label, category):
    '''label is in the following order:
        ['filename', ['object', 'object', ...], [xmin, xmin, ...], 
        [ymin, ymin, ...], [xmax, xmax, ...], [ymax, ymax, ...]]
    '''
    file_name, obj_names, xmins, ymins, xmaxs, ymaxs = label
    img_str        = cv2.imencode('.jpg', img_array)[1].tobytes()
    height, width  = img_array.shape[:2]
    filename       = bytes(file_name, 'utf-8')
    img_format     = b'jpeg'
    key            = hashlib.sha256(img_str).hexdigest()
    
#     xmins = map(int, xmins)
#     ymins = map(int, ymins)
#     xmaxs = map(int, xmaxs)
#     ymaxs = map(int, ymaxs)
    
    xmins          = [xmin / width for xmin in xmins]
    ymins          = [ymin / height for ymin in ymins]
    xmaxs          = [xmax / width for xmax in xmaxs]
    ymaxs          = [ymax / height for ymax in ymaxs]
    classes_text   = [bytes(obj_name, 'utf-8') for obj_name in obj_names]
    classes        = [CATEGORY[obj_name] for obj_name in obj_names]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/filename': _bytes_feature(filename),
        'image/source_id': _bytes_feature(filename),
        'image/key/sha256': _bytes_feature(key.encode('utf-8')),
        'image/encoded': _bytes_feature(img_str),
        'image/format': _bytes_feature(img_format),
        'image/object/bbox/xmin': _float_list_feature(xmins),
        'image/object/bbox/ymin': _float_list_feature(ymins),
        'image/object/bbox/xmax': _float_list_feature(xmaxs),
        'image/object/bbox/ymax': _float_list_feature(ymaxs),
        'image/object/class/text': _bytes_list_feature(classes_text),
        'image/object/class/label': _int64_list_feature(classes)
    }))
    
    return tf_example


def stdout_write(message):
    sys.stdout.write('\r')
    sys.stdout.write(message)
    sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path')
    parser.add_argument('--csv_path')
    parser.add_argument('--out_path')
    args = parser.parse_args()
    csv = pd.read_csv(args.csv_path)

    with tf.python_io.TFRecordWriter(args.out_path) as writer:
        groupby_filename = csv.groupby('filename')
        for i, (filename, data) in enumerate(groupby_filename,1):
            img = cv2.imread(str(Path(args.img_path, filename)))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label = [filename] + [list(a) for a in (data['obj_name'],
                                                    data['xmin'], data['ymin'],
                                                    data['xmax'], data['ymax'])]
            
            tf_example = make_tfexample(img, label, CATEGORY)
            
            writer.write(tf_example.SerializeToString())
            message = 'Converted {} / {} images&labels to tfrecord'.format(i, len(groupby_filename))
            stdout_write(message)
    print('\ndone!')
