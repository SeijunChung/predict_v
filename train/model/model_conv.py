import tensorflow as tf
import tensorflow.keras as tk

# For pycharm autocomplete issue
import typing
if typing.TYPE_CHECKING:
    from keras.api._v2 import keras as tk


class myAttention(tk.layers.Layer):
    def __init__(self, units, name="attention"):
        super(myAttention, self).__init__(name=name)
        self.units = units
        self.dense1 = tk.layers.Dense(16)
        self.dense2 = tk.layers.Dense(16)
        self.w_e = tk.layers.Dense(units, use_bias=False)
        self.u_e = tk.layers.Dense(units, use_bias=False)
        self.v_e = tk.layers.Dense(1, use_bias=False)
        self.relu = tk.layers.ReLU()
        self.tanh = tk.layers.Activation('tanh')
        self.softmax = tk.layers.Softmax()
        self.output_time_steps = 10

    def call(self, query, feature, training=False):
        '''
        :param query: encoded past information for query, shape: [batch, self.output_time_steps, m+self.output_time_steps]
        :param feature: feature map for key, shape: [batch, self.output_time_steps, feature_length, n]
        :return: input attention weights, shape: [batch, feature_length, self.output_time_steps]
        '''
        feature = self.relu(self.dense2(self.relu(self.dense1(feature))))
        v1 = self.u_e(feature)  # shape: [batch, self.output_time_steps, feature_length, units]
        v2 = self.w_e(query)  # shape: [batch, self.output_time_steps, units]
        v2 = tf.expand_dims(v2, axis=2)  # shape: [batch, self.output_time_steps, 1, units]
        e_t = tf.squeeze(self.v_e(self.tanh(v1 + v2)), axis=-1)  # shape: [batch, self.output_time_steps, feature_length]
        alpha_t = self.softmax(e_t)  # shape: [batch, self.output_time_steps, feature_length]
        return alpha_t

    def get_config(self):
        return {'w': self.w_e,
                'u': self.u_e,
                'v': self.v_e,
                'units': self.units,
                'name': self.name}


class Encoder_convs(tk.Model):
    def __init__(self, filters=[32, 32, 32, 5], sizes=[3, 3, 3, 1], strides=[1, 1, 1, 1], name="enc_convs"):
        super(Encoder_convs, self).__init__(name=name)
        self.filters = filters
        self.sizes = sizes
        self.strides = strides
        # Layers
        self.convs = []
        first = True
        for fs, ks, ss in zip(filters, sizes, strides):
            if first:
                self.convs.append(tk.layers.Conv1D(fs, ks, ss, padding='valid', input_shape=(20,6)))
                first=False
            else:    
                self.convs.append(tk.layers.Conv1D(fs, ks, ss, padding='valid'))
        self.relu = tk.layers.ReLU()
#         self.relu = tk.layers.LeakyReLU(alpha=0.2)

    def call(self, x, training=False):
        # forward
        for i in range(len(self.filters)):
            x = self.convs[i](x)
            x = self.relu(x)

        return x

    def get_config(self):
        return {'convs': self.convs,
                'filters': self.filters,
                'kernel_sizes': self.sizes,
                'strides': self.strides,
                'name': self.name}


class conv1d_mlp(tk.Model):
    def __init__(self, past_filters=[32, 32, 32], past_sizes=[3, 3, 3], past_strides=[1, 1, 1],
                 forward_filters=[32, 32, 32], forward_sizes=[3, 3, 3], forward_strides=[2, 2, 2],
                 name="conv_mlp"):
        super(conv1d_mlp, self).__init__(name=name)
        # sub models and layers
        self.enc_past = Encoder_convs(past_filters, past_sizes, past_strides, name="enc_past")
        self.enc_forward = Encoder_convs(forward_filters, forward_sizes, forward_strides, name="enc_forward")
        self.dense1 = tk.layers.Dense(32)
        self.dense2 = tk.layers.Dense(32)
        self.dense3 = tk.layers.Dense(32)
        self.dense4 = tk.layers.Dense(10)
        self.relu = tk.layers.ReLU()
        self.tanh = tk.layers.Activation('tanh')
        self.flatten = tk.layers.Flatten()
        self.concat = tk.layers.Concatenate(axis=-1)

    def call(self, x_past, x_forward, training=False):
        code_past = self.flatten(self.enc_past(x_past))
        code_past = self.relu(self.dense1(code_past))
        code_forward = self.flatten(self.enc_forward(x_forward))
        code_forward = self.relu(self.dense2(code_forward))
        code = self.concat([code_past, code_forward])
        x = self.relu(self.dense3(code))
        x = self.tanh(self.dense4(x))
        output = tf.reshape(x, (-1, 10, 1)) + x_past[:, -1:, -1:]

        return output

    def get_config(self):
        return {'encoder_past': self.enc_past,
                'encoder_forward': self.enc_forward,
                'dense': [self.dense1, self.dense2, self.dense3, self.dense4]}

    
class conv1d_att(tk.Model):
    def __init__(self, past_filters=[32, 32, 32], past_sizes=[3, 3, 3], past_strides=[1, 1, 1],
                 forward_filters=[32, 32, 32], forward_sizes=[3, 3, 3], forward_strides=[2, 2, 2],
                 name="conv_att"):
        super(conv1d_att, self).__init__(name=name)
        # sub models and layers
        self.enc_past = Encoder_convs(past_filters, past_sizes, past_strides, name="enc_past")
        self.spatial_att = myAttention(16, "spatial_att")
        self.enc_forward = Encoder_convs(forward_filters, forward_sizes, forward_strides, name="enc_forward")
        self.dense1 = tk.layers.Dense(32)
        self.dense2 = tk.layers.Dense(32)
        self.dense3 = tk.layers.Dense(32)
        self.dense4 = tk.layers.Dense(32)
        self.dense5 = tk.layers.Dense(1)
        self.relu = tk.layers.ReLU()
        self.tanh = tk.layers.Activation('tanh')
        self.output_time_steps = 10
        self.flatten = tk.layers.Flatten()

    def call(self, x_past, x_forward, pos_enc_past, pos_enc_fwd, training=False):
        x_forward = tf.expand_dims(x_forward, axis=1)
        pos_enc_fwd = tf.expand_dims(pos_enc_fwd, axis=1)
        code_past = self.relu(self.dense1(self.flatten(self.enc_past(x_past))))
        code_past = tf.tile(tf.expand_dims(code_past, axis=1), [1, self.output_time_steps, 1])
        query = self.relu(self.dense2(tf.concat([code_past, pos_enc_past], axis=-1)))  # shape: [-1, self.output_time_steps, k+self.output_time_steps]
        alpha = self.spatial_att(query, tf.concat([x_forward, pos_enc_fwd],axis=-1))  # shape: [-1, self.output_time_steps, forward_length]
        att_feature = tf.expand_dims(alpha, axis=3) * x_forward  # shape: [-1, self.output_time_steps, forward_length, forward_channel]
        code = self.relu(self.dense3(tf.reshape(self.enc_forward(att_feature), [-1, self.output_time_steps, 94 * 32])))  # shape: [-1, self.output_time_steps, m]
        code = tf.concat([query, code], axis=-1)  # shape: [-1, self.output_time_steps, k+m]
        x = self.relu(self.dense4(code))  # shape: [-1, self.output_time_steps, 1]
        output = self.tanh(self.dense5(x)) + x_past[:, -1:, -1:]

        return output

    def get_config(self):
        return {'encoder_past': self.enc_past,
                'spatial_att': self.spatial_att,
                'encoder_forward': self.enc_forward,
                'dense': [self.dense1, self.dense2, self.dense3, self.dense4, self.dense5]}



'''
# Test Code
import numpy as np
def pos_encoder(pos, freq=400, d=4):
    pos_enc_fwd = np.zeros([pos.shape[0], d], dtype=np.float32)
    for i in range(d):
        if i % 2 == 0:
            pos_enc_fwd[:, i] = np.sin(pos / (freq ** (i / d)))
        else:
            pos_enc_fwd[:, i] = np.cos(pos / (freq ** (i / d)))
    return pos_enc_fwd

past_input_sample = tf.random.normal((2, 20, 1))
forward_input_sample = tf.random.normal((2, 100, 4))
# forward_input_sample = tf.tile(tf.expand_dims(forward_input_sample, axis=1), [1, 10, 1, 1])
# pos_enc = tf.eye(10, 10, batch_shape=[2])
# pos_enc = tf.tile(tf.expand_dims(tf.range(10), axis=0), [2, 1])
pos_enc_past = tf.tile(tf.expand_dims(pos_encoder(np.arange(10), 40, 32), axis=0), [2, 1, 1])
pos_enc_fwd = tf.tile(tf.expand_dims(pos_encoder(np.arange(100), d=32), axis=0), [2, 1, 1])
model = conv1d_att_w_pos_fwd()
output = model(past_input_sample, forward_input_sample, pos_enc_past, pos_enc_fwd)
# spatial_att_layer_1 = InputAttention(units=7)
# spatial_att_layer_2 = Attention(units=5)
# channel_att_layer = Attention(units=3)
# a, query = spatial_att_layer_1(past_input_sample, forward_input_sample)
# b = tf.expand_dims(a, axis=3)*tf.expand_dims(forward_input_sample, axis=1)
# c = channel_att_layer(query, tf.transpose(b, [0, 1, 3, 2]))
# d = tf.expand_dims(c, axis=2)*b
# e = spatial_att_layer_2(query, d)
print(output)
'''