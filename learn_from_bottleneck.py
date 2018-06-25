
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, BatchNormalization, GlobalAveragePooling2D, Dropout, LeakyReLU
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import l2

import argparse
import os

def bottleneck_model(input_shape):
    num_classes = 133    

    bottleneck_model = Sequential()
    #bottleneck_model.add(Conv2D(128, kernel_size=(3,3), padding='same', input_shape=input_shape))
    #bottleneck_model.add(Activation('relu'))
    #bottleneck_model.add(BatchNormalization());
    #bottleneck_model.add(Dropout(0.5))

    #bottleneck_model.add(Conv2D(256, kernel_size=(3,3), strides=(2, 2), padding='same'))
    #bottleneck_model.add(Activation('relu'))
    #bottleneck_model.add(BatchNormalization());
    #bottleneck_model.add(Dropout(0.5))

    #bottleneck_model.add(Flatten()) 
        
    bottleneck_model.add(GlobalAveragePooling2D(input_shape=input_shape))
    
    bottleneck_model.add(Dense(256, kernel_regularizer=l2(0.001)))
    #bottleneck_model.add(Activation('relu'))
    bottleneck_model.add(LeakyReLU(alpha=0.3))
    bottleneck_model.add(BatchNormalization());
    bottleneck_model.add(Dropout(0.5))
    
    bottleneck_model.add(Dense(128, kernel_regularizer=l2(0.001)))
    #bottleneck_model.add(Activation('relu'))
    bottleneck_model.add(LeakyReLU(alpha=0.3))
    bottleneck_model.add(BatchNormalization());
    bottleneck_model.add(Dropout(0.5))

    bottleneck_model.add(Dense(num_classes))
    bottleneck_model.add(Activation('softmax'))

    bottleneck_model.summary()

    return bottleneck_model

if __name__ == '__main__':

    DIR_SAVED_MODELS = 'saved_models'

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_features", type=str,
                        help="directory to image set (contains train, valid, test folder")

    parser.add_argument("--file_labels", type=str, 
                        help="the base net that will be used for feature extraction")

    parser.add_argument("--model_name", type=str, 
                        help="the model name that will be used in the weights file name: weights.best.<model_name>.hdf5")

    # parser.add_argument("-v", "--verbose", action="store_true",
    #                     help="increase output verbosity")

    args = parser.parse_args()


    # fn_labels = 'bottleneck_features/labels_train_val_test.npz'
    # fn_bottleneck_features = 'bottleneck_features/bottleneck_train_valid_test_inceptionV3.npz'

    fn_labels = args.file_labels
    fn_bottleneck_features = args.file_features
    model_name = args.model_name

    labels_data = np.load(fn_labels)
    y_train = labels_data['y_train'] * 0.9
    y_valid = labels_data['y_valid']
    y_test = labels_data['y_test']

    print(y_train.shape)

    feature_data = np.load(fn_bottleneck_features)
    
    # extract train, valid, and test features from 
    bottleneck_features_train = feature_data['bottleneck_features_train']
    bottleneck_features_valid = feature_data['bottleneck_features_valid']
    bottleneck_features_test = feature_data['bottleneck_features_test']

    model = bottleneck_model(bottleneck_features_train.shape[1:])
    # For a multi-class classification problem
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    epochs = 100
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, 
        patience=3, min_lr=0.00001, verbose=1)

    if not os.path.exists(DIR_SAVED_MODELS):
        os.makedirs(DIR_SAVED_MODELS)

    checkpoint = ModelCheckpoint(DIR_SAVED_MODELS + '/weights.best.{0}.hdf5'.format(model_name), verbose=1, save_best_only=True)

    model.fit(bottleneck_features_train, y_train, epochs=epochs, batch_size=32, verbose=1,
        validation_data=(bottleneck_features_valid, y_valid), callbacks=[reduce_lr, checkpoint])

    predictions = [np.argmax(model.predict(np.expand_dims(feature, axis=0))) for feature in bottleneck_features_test]

    # report test accuracy
    test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(y_test, axis=1))/len(predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)

