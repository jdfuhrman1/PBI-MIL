import numpy as np
import os
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from skimage.measure import block_reduce
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from scipy import ndimage

def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
        return v
    return v/norm

def pool(arr,func):
    '''
    Downsamples an array by pooling
    arr: Input array, should have 4 dimensions
    func: Pooling function to use, e.g. np.max or np.mean
    '''
    s1,s2,s3,s4 = arr.shape
    return block_reduce(arr, block_size=(1,s2,s3,1), func=func).reshape(1,-1)


def poolFeats(path, savePath):

    print('Extracting features with ResNet50 to be saved in', savePath)
    model = ResNet50(include_top=False,weights='imagenet')
    pooledFeats = K.function([model.layers[0].input],
                             [model.layers[6].output,
                                model.layers[38].output,
                                model.layers[80].output,
                                model.layers[142].output,
                                model.layers[174].output])
    
    directories = [f for f in os.listdir(path)]
    directories.sort()

    for counter, d in enumerate(directories):
        print('On case ', d, ' of ', str(len(directories)))
        myarray = []
        filenames = os.listdir(os.path.join(path, d))
        filenames.sort()
        for i, k in enumerate(filenames):
            X = image.img_to_array(image.load_img(os.path.join(path,d, k)))
            x = np.expand_dims(X, axis = 0)
            layerOutputs = pooledFeats([x])
            l0 = normalize(pool(layerOutputs[0],np.mean))
            l1 = normalize(pool(layerOutputs[1],np.mean))
            l2 = normalize(pool(layerOutputs[2],np.mean))
            l3 = normalize(pool(layerOutputs[3],np.mean))
            l4 = normalize(pool(layerOutputs[4],np.mean))
            feats = normalize(np.concatenate((l0,l1,l2,l3,l4),axis=1))
            for corrElements in range(np.shape(feats)[1]):
                if feats[0,corrElements] > 100.0:
                    feats[0,corrElements] = 0.0
            myarray.append(feats)
        np.savetxt(savePath + d, np.concatenate(myarray,axis=0))
    return
