
import os
import numpy as np

from scipy.ndimage import imread
from scipy.misc import imresize
from tqdm import tqdm

import random

import cv2
import multiprocessing

def read_images_filenames(dir_images, folders, extension='jpg'):
    """Read all jpg files in a given directory."""
    image_files = []
    image_labels = []

    for folder in folders:
        # read training filenames
        current_folder = dir_images + folder +'/'
        folder_contents = os.listdir(current_folder)
                
        for entry in folder_contents:
            if os.path.isfile(os.path.join(current_folder, entry)):
                if entry.endswith(extension):
                    image_files.append(current_folder + entry)
                    
        image_files.sort()
        for fn in image_files:
            image_labels.append(folder)            

    return image_files, image_labels

def read_subfolders(directory):
    contents = os.listdir(directory)
    folders = []
    for entry in contents:
        if os.path.isdir(os.path.join(directory, entry)):
            folders.append(entry)

    return folders

def random_crop_from_tensor(img, crop_size):

    diff_x = img.shape[2] - crop_size
    diff_y = img.shape[1] - crop_size

    offset_x = random.randrange(diff_x)
    offset_y = random.randrange(diff_y)

    crop = img[0, offset_y:offset_y+crop_size, offset_x:offset_x+crop_size, :]
    #print('crop {0}'.format(crop.shape))
    return crop

image_cache = dict()

# thread that preload the images
def random_crop_gen(images_filename, labels, batch_size, target_width, net_width, num_channels, test=False, preprocess_func=None):

    if (preprocess_func == None):
        print('No preprocessing function was provided')

    #random.shuffle(image_list)
    X = np.zeros((batch_size, net_width, net_width, num_channels), dtype=np.float)
    Y = np.zeros((batch_size, 133), dtype=np.float)
    #for image_idx in image_list:   
    iter_idx = 0     
    while True:      
        for idx in range(batch_size):            
            image_idx = random.randrange(0,len(images_filename))    
            if not images_filename[image_idx] in image_cache:
            
                img = imread(images_filename[image_idx], mode='RGB') 
                
                height = img.shape[0]
                width = img.shape[1]    
                #print(' orig {0}x{1}'.format(width, height))    
                
                if height < width:
                    scale_h = target_width / height
                    new_width = width * scale_h
                    new_height = height * scale_h

                    new_width = int(new_width + 0.5)
                    new_height = int(new_height + 0.5)

                    #print('{0}x{1}'.format(new_width, new_height))
                        
                else:
                    
                    scale_w = target_width / width
                    new_width = width * scale_w
                    new_height = height * scale_w

                    new_width = int(new_width + 0.5)
                    new_height = int(new_height + 0.5)

                    #print('{0}x{1}'.format(new_width, new_height))

                #print('{0}x{1}'.format(new_width, new_height))
                assert(new_width >= target_width and new_height >= target_width)
                
                img = imresize(img, (new_height, new_width))
                img = np.expand_dims(img, 0)

                image_cache[images_filename[image_idx]] = img
            else:
                img = image_cache[images_filename[image_idx]]
            # create random crops
            crop = random_crop_from_tensor(img, net_width)

            # cv2.imshow('img', crop)
            # cv2.waitKey(0)
            X[idx, :, :, :] = crop.astype(np.float)
            Y[idx, :] = labels[image_idx]

        if preprocess_func is not None:
            yield preprocess_func(X), Y
        else:
            yield X, Y

def read_images_to_tensor(image_files, target_width):
        
    image_size = 224
    image_tensor = np.zeros((len(image_files), image_size, image_size, 3), dtype=np.float)

    for index, fn in enumerate(image_files):
        img = imread(image_files[index], mode='RGB') 
        try:              
                        
            img = imresize(img, (target_width, target_width))                        
            img = np.expand_dims(img, 0)  
            image_tensor[index, :, :, :] = img.astype(np.float)

        except:
            print(image_files[index])

    
    return image_tensor