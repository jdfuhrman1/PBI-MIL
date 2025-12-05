from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Dense, Dropout, multiply, Concatenate
from .custom_layers import Mil_Attention, Last_Sigmoid



def feature_net():

    weight_decay = 0.0
    useGated = True

    #obtain inputs
    feats = Input((3904), name='image')
    pos = Input((1), name='position')
    
    feats = Concatenate()([feats, pos])
    fc1 = Dense(512, activation='relu', kernel_initializer='he_normal', name='fc1', kernel_regularizer=l2(weight_decay))(feats)
    fc1 = Dropout(0.5)(fc1)
    fc2 = Dense(512, activation='relu', kernel_initializer='he_normal', name='fc2', kernel_regularizer=l2(weight_decay))(fc1)
    fc2 = Dropout(0.5)(fc2)
    
    alpha = Mil_Attention(L_dim=128, output_dim=1, name='alpha', use_gated=useGated, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(fc2)
    x_mul = multiply([alpha, fc2])
    
    out = Last_Sigmoid(output_dim=1, name='FC1_sigmoid', kernel_initializer='he_normal', includeGCS=False, kernel_regularizer=l2(weight_decay))(x_mul)

    model = Model(inputs=[inp, pos], outputs=[out])

    return model
    




