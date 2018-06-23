
import os
from sklearn.preprocessing import LabelBinarizer

from scipy.ndimage import imread
from scipy.misc import imresize

import matplotlib.pyplot as plt
import numpy as np

from keras.models import Model

import image_utils
import argparse

def create_bottlenet_features_inception_v3(image_tensor):
        
    from keras.applications import InceptionV3
    from keras.applications import preprocess_input as preprocess_input_inceptionV3

    inceptionV3 = InceptionV3(weights='imagenet', include_top=False)
    print(inceptionV3.summary())

    bottleneck_model = Model(inputs=inceptionV3.input, outputs=inceptionV3.get_layer('mixed').output)

    print(bottleneck_model.summary())

    bottleneck_features = bottleneck_model.predict(preprocess_input_inceptionV3(image_tensor))
    print(bottleneck_features.shape)

    return bottleneck_features

def create_bottlenet_features_vgg19(image_tensor):

    from keras.applications.vgg19 import VGG19
    from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
    vgg19 = VGG19(weights='imagenet', include_top=False)
    print(vgg19.summary())

    bottleneck_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_pool').output)

    print(bottleneck_model.summary())

    bottleneck_features = bottleneck_model.predict(preprocess_input_vgg19(image_tensor))
    print(bottleneck_features.shape)

    return bottleneck_features

def create_bottlenet_features_xception(image_tensor):

    from keras.applications.xception import Xception
    from keras.applications.xception import preprocess_input as preprocess_input_xception
    xception = Xception(weights='imagenet', include_top=False)
    print(xception.summary())

    bottleneck_model = Model(inputs=xception.input, outputs=xception.get_layer('block14_sepconv2_act').output)

    print(bottleneck_model.summary())

    bottleneck_features = bottleneck_model.predict(preprocess_input_xception(image_tensor))
    print(bottleneck_features.shape)

    return bottleneck_features

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=str,
                        help="directory to image set (contains train, valid, test folder")

    parser.add_argument("target_width", type=int,
                        help="target width for the base network")

    parser.add_argument("base_net", type=str, 
                        help="the base net that will be used for feature extraction")
    # parser.add_argument("-v", "--verbose", action="store_true",
    #                     help="increase output verbosity")
    args = parser.parse_args()

    methods = dict()
    methods['vgg19']  = True
    methods['inceptionV3']  = True
    methods['xception']  = True    

    if not os.path.exists(args.base_dir):
        print('base_dir {0} does not exit'.format(args.base_dir))
        exit()
    
    base_dir = args.base_dir

    target_width = args.target_width
    base_net = args.base_net

    if not base_net in methods:
        print('unknown base net')
        exit()    
    
    dir_training_images = base_dir + 'train/'
    dir_valid_images = base_dir + 'valid/'
    dir_test_images = base_dir + 'test/'

    assert(os.path.exists(dir_training_images))     
    assert(os.path.exists(dir_valid_images))
    assert(os.path.exists(dir_test_images))

    dir_bottleneck_features = 'bottleneck_features'
    if not os.path.exists(dir_bottleneck_features):
        os.makedirs(dir_bottleneck_features)

    # read train folders    
    train_folders = image_utils.read_subfolders(dir_training_images)    
    print('Number of folder in train set: {0}'.format(len(train_folders)))
    valid_folders = image_utils.read_subfolders(dir_valid_images)
    print('Number of folder in valid set: {0}'.format(len(valid_folders)))
    test_folders = image_utils.read_subfolders(dir_test_images)
    print('Number of folder in test set: {0}'.format(len(test_folders)))

    train_folders.sort()
    valid_folders.sort()
    test_folders.sort()
    
    train_image_files, train_image_labels = image_utils.read_images_filenames(dir_training_images, train_folders)    
    print('Number of training images: {0}'.format(len(train_image_files)))
    valid_image_files, valid_image_labels = image_utils.read_images_filenames(dir_valid_images, valid_folders)    
    print('Number of validation images: {0}'.format(len(valid_image_files)))
    test_image_files, test_image_labels = image_utils.read_images_filenames(dir_test_images, test_folders)    
    print('Number of test images: {0}'.format(len(test_image_files)))

    y_train = LabelBinarizer().fit_transform(train_image_labels)
    y_valid = LabelBinarizer().fit_transform(valid_image_labels)

    #img = imread(train_image_files[0], mode='RGB')

    # image size VGG, RESNET: 224
    # image size XCeption, Inception 299
    image_tensor_train = image_utils.read_images_to_tensor(train_image_files, target_width=target_width)
    image_tensor_valid = image_utils.read_images_to_tensor(valid_image_files, target_width=target_width)
    image_tensor_test = image_utils.read_images_to_tensor(test_image_files, target_width=target_width)
    
    bottleneck_features_train = create_bottlenet_features_xception(image_tensor_train)
    bottleneck_features_valid = create_bottlenet_features_xception(image_tensor_valid)
    bottleneck_features_test = create_bottlenet_features_xception(image_tensor_test)

    # # Predict
    np.savez( dir_bottleneck_features + '/bottleneck_train_valid_test_xception.npz',
        bottleneck_features_train=bottleneck_features_train,
        bottleneck_features_valid=bottleneck_features_valid,
        bottleneck_features_test=bottleneck_features_test)

    np.savez( dir_bottleneck_features + '/labels_train_val_xception.npz',
        y_train=y_train,
        y_valid=y_valid)    