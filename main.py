'''
Written to demonstrate use of the PBI-MIL method described in the manuscript
"AI-Aided Triage for GSWH: Validating an Interpretable HCT-Based Mortality Model"

Note, this can not be run without significant edits/additions, particularly those
regarding loading in data.  This gives a basic framework, but is not general or complete.
'''

import random
import numpy as np
import tensorflow as tf

from utl.PoolFeats import poolFeats
from utl.datagen import data_gen
from utl import MIL_Net
from utl.metrics import bag_loss, bag_accuracy
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


fit = False
lr = 1e-4
batch_size = 1
model_type = 'transfer' #one of '3D', 'transfer', and 'extracted'

#Define file paths, hyperparameters, etc.
img_path = '/'      #path for image files, saved as greyscale images, formatted as case directories with images contained within.
feats_path = '/'    #path to save extracted features

poolFeats(img_path, feats_path) #Current version utilizes ResNet50 backbone with pre-trained ImageNet weights.  Note, this function has NOT been written for general use

for imodelNum in range(100):    
    model_name = '.h5'
    
    np.random.seed(imodelNum)

    #obtain dicts for train/val/test sets.  In this example, dict indices are case IDs which have path names to previously extracted slice feature files (acquired in poolFeats function) and the case label.  This is inefficient, rewrite at a later date.
    tr_dict = {}
    va_dict = {}
    te_dict = {}

    train_gen = data_gen(tr_dict, batch_size)
    val_gen = data_gen(va_dict, batch_size)
    
    #Training Process
    model = MIL_Net.feature_net()
    model.summary()
    model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss=bag_loss, metrics=[bag_accuracy, tf.keras.metrics.AUC()])

    if fit:
        checkpoint_fixed_name = ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        EarlyStop = EarlyStopping(monitor='val_loss', patience=30, verbose=1)
        Reducer = ReduceLROnPlateau(factor=0.1, patience=16, min_lr=0.0000001, verbose=1)
        callbacks = [checkpoint_fixed_name, EarlyStop, Reducer]
        history = model.fit_generator(generator=train_gen, steps_per_epoch=len(tr_dict)/batch_size, epochs=200, validation_data=val_gen, validation_steps=len(va_dict)/batch_size, callbacks=callbacks)

    model.load_weights(model_name)

    #Testing process
    preds = []
    runBags = K.function([model.layers[0].input, model.layers[1].input],  #input images in [0] and position in [1]; replace with position embedding later
                         [model.layers[7].output,       #for accessing attention values
                          model.layers[8].output,       #for accessing embeddings
                          model.layers[9].output])      #for accessing predictions

    index = list(te_dict.keys())
    for i in range(len(te_dict)):
    
        batch_feats = np.loadtxt(te_dict[index[i]][0], dtype=np.float32)
        pos = np.zeros([np.shape(batch_feats)[0],1])
        for j in range(np.shape(batch_feats)[0]):
            pos[j,0] = j/np.shape(batch_feats)[0]
        x = {"image":batch_feats, "position":pos}
        outputs = runBags([x['image'], x['position']])
        preds.append(outputs[2])
            

