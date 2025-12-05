from tensorflow.keras.layers import Layer, Dense, LayerNormalization
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, regularizers
import tensorflow as tf


def attention_entropy(attn_weights):
    epsilon = 1e-8
    entropy = -tf.reduce_sum(attn_weights * tf.math.log(attn_weights + epsilon), axis=0)
    return tf.reduce_mean(entropy)

class MultiHeadGatedAttention(Layer):
    def __init__(self, num_heads=4, hidden_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.attn_layers = []  # One set of weights per head

        for _ in range(num_heads):
            self.attn_layers.append({
                'V': Dense(hidden_dim, activation='tanh'),
                'U': Dense(hidden_dim, activation='sigmoid'),
                'W': Dense(1, use_bias=False)
            })

    def call(self, inputs, training=False):
        # inputs: shape (batch_size, num_instances, feature_dim)
        head_outputs = []

        for i in range(self.num_heads):
            V = self.attn_layers[i]['V'](inputs)  # (B, N, H)
            U = self.attn_layers[i]['U'](inputs)  # (B, N, H)
            H = tf.multiply(V, U)                # (B, N, H)
            logits = self.attn_layers[i]['W'](H)  # (B, N, 1)
            attn_weights = tf.nn.softmax(logits, axis=1)  # (B, N, 1)
            weighted_sum = tf.reduce_sum(attn_weights * inputs, axis=1)  # (B, D)
            head_outputs.append(weighted_sum)

        # Combine outputs: (B, num_heads * feature_dim)
        return tf.concat(head_outputs, axis=-1)  # or tf.reduce_mean([...], axis=0)

class Mil_Attention(Layer):
    """
    Mil Attention Mechanism

    This layer contains Mil Attention Mechanism

    # Input Shape
        2D tensor with shape: (batch_size, input_dim)

    # Output Shape
        2D tensor with shape: (1, units)
    """

    def __init__(self, L_dim, output_dim, kernel_initializer='glorot_uniform', kernel_regularizer=None,
                    use_bias=True, use_gated=False, **kwargs):
        self.L_dim = L_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.use_gated = use_gated

        self.v_init = initializers.get(kernel_initializer)
        self.w_init = initializers.get(kernel_initializer)
        self.u_init = initializers.get(kernel_initializer)


        self.v_regularizer = regularizers.get(kernel_regularizer)
        self.w_regularizer = regularizers.get(kernel_regularizer)
        self.u_regularizer = regularizers.get(kernel_regularizer)

        super(Mil_Attention, self).__init__(**kwargs)

    def build(self, input_shape):

        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.V = self.add_weight(shape=(input_dim, self.L_dim),
                                      initializer=self.v_init,
                                      name='v',
                                      regularizer=self.v_regularizer,
                                      trainable=True)


        self.w = self.add_weight(shape=(self.L_dim, 1),
                                    initializer=self.w_init,
                                    name='w',
                                    regularizer=self.w_regularizer,
                                    trainable=True)


        if self.use_gated:
            self.U = self.add_weight(shape=(input_dim, self.L_dim),
                                     initializer=self.u_init,
                                     name='U',
                                     regularizer=self.u_regularizer,
                                     trainable=True)
        else:
            self.U = None

        self.input_built = True
        #self.layer_norm = LayerNormalization()


    def call(self, x, mask=None):
        n, d = x.shape
        ori_x = x
        # do Vhk^T
        x = K.dot(x, self.V) # (2,64)
        #x = self.layer_norm(x) #Added this line!
        x = K.tanh(x)
        
        if self.use_gated:
            #gate_x = K.sigmoid(K.dot(ori_x, self.U))
            gate_x = K.dot(ori_x, self.U)
            #gate_x = self.layer_norm(gate_x) #Added this line!
            gate_x = K.sigmoid(gate_x)
            ac_x = x * gate_x
        else:
            ac_x = x

        # do w^T x
        soft_x = K.dot(ac_x, self.w)  # (2,64) * (64, 1) = (2,1)
        #soft_x = K.tanh(soft_x) #Added this line!!
        alpha = K.softmax(K.transpose(soft_x)) # (2,1)
        alpha = K.transpose(alpha)
        #self.add_loss(0.0001*attention_entropy(alpha))
        return alpha

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'v_initializer': initializers.serialize(self.V.initializer),
            'w_initializer': initializers.serialize(self.w.initializer),
            'v_regularizer': regularizers.serialize(self.v_regularizer),
            'w_regularizer': regularizers.serialize(self.w_regularizer),
            'use_bias': self.use_bias
        }
        base_config = super(Mil_Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Last_Sigmoid(Layer):
    """
    Attention Activation

    This layer contains a FC layer which only has one neural with sigmoid actiavtion
    and MIL pooling. The input of this layer is instance features. Then we obtain
    instance scores via this FC layer. And use MIL pooling to aggregate instance scores
    into bag score that is the output of Score pooling layer.
    This layer is used in mi-Net.

    # Arguments
        output_dim: Positive integer, dimensionality of the output space
        kernel_initializer: Initializer of the `kernel` weights matrix
        bias_initializer: Initializer of the `bias` weights
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix
        bias_regularizer: Regularizer function applied to the `bias` weights
        use_bias: Boolean, whether use bias or not
        pooling_mode: A string,
                      the mode of MIL pooling method, like 'max' (max pooling),
                      'ave' (average pooling), 'lse' (log-sum-exp pooling)

    # Input shape
        2D tensor with shape: (batch_size, input_dim)
    # Output shape
        2D tensor with shape: (1, units)
    """
    def __init__(self, output_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    kernel_regularizer=None, bias_regularizer=None,
                    use_bias=True, includeGCS=False, **kwargs):
        self.output_dim = output_dim

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.use_bias = use_bias
        self.includeGCS = includeGCS
        super(Last_Sigmoid, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.includeGCS:
            assert len(input_shape[0]) == 2
            input_dim = input_shape[0][1]
            
        else:
            assert len(input_shape) == 2
            input_dim = input_shape[1]
        if self.includeGCS:
            self.kernel = self.add_weight(shape=(input_dim+1, self.output_dim),
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer)
        else:
            self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer)
        else:
            self.bias = None

        self.input_built = True

    def call(self, inputs):
        if self.includeGCS:
            x, y = inputs
        else:
            x = inputs
        n, d = x.shape
        x = K.sum(x, axis=0, keepdims=True)
        
        if self.includeGCS:
            x = K.concatenate([x,y], axis=1)
        
        # compute instance-level score
        x = K.dot(x, self.kernel)

        if self.use_bias:
            x = K.bias_add(x, self.bias)

        # sigmoid
        out = K.sigmoid(x)


        return out

    def compute_output_shape(self, input_shape):
        if self.includeGCS:
            shape = list(input_shape[0])
        else:
            shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'kernel_initializer': initializers.serialize(self.kernel.initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'use_bias': self.use_bias
        }
        base_config = super(Last_Sigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

