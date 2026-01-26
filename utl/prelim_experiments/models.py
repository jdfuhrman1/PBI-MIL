from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv3D, MaxPooling3D, BatchNormalization, GlobalAveragePooling3D, Add, Concatenate


def conv_block(inp, n): #Add residual connection here if desired
    x = Conv3D(n, kernel_size=(3,3,3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), padding='same')(inp)
    x = Conv3D(n, kernel_size=(3,3,3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), padding='same')(x)
    x = MaxPooling3D((2,2,2))(x)
    x = BatchNormalization()(x)

def net_3d(transfer=True, fullthreeD=False, backbone='VGG19', includeGCS=False):
    
    weight_decay = 0.0
    n = 16
    num_blocks = 4
    
    x = Input((32,224,224,1), name='image') #z-axis either padded or cropped/interpolated to 32 slices
    for i in range(num_blocks):
        x = conv_block(x, (2**i)*n)
        
    #feats = Flatten()(p3)
    #fc1 = Dense(512, activation='relu', kernel_initializer='he_normal', name='fc1', kernel_regularizer=l2(weight_decay))(feats)
    fc1 = GlobalAveragePooling3D()(p4)
    fc1 = Dropout(0.5)(fc1)
    fc2 = Dense(512, activation='relu', kernel_initializer='he_normal', name='fc2', kernel_regularizer=l2(weight_decay))(fc1)
    fc2 = Dropout(0.5)(fc2)
    out = Dense(1, activation='sigmoid', kernel_initializer='he_normal', name='classifier')(fc2)
        
    model = Model(inputs=[inp], outputs=[out])
    return model
    




