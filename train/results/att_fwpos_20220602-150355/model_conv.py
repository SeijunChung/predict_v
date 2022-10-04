import tensorflow as tf
import tensorflow.keras as tk
import random
import time

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
        v1 = self.u_e(feature) # shape: [batch, self.output_time_steps, feature_length, units]
        v2 = self.w_e(query) # shape: [batch, self.output_time_steps, units]
        v2 = tf.expand_dims(v2, axis=2) # shape: [batch, self.output_time_steps, 1, units]
        e_t = tf.squeeze(self.v_e(self.tanh(v1+v2)), axis=-1) # shape: [batch, self.output_time_steps, feature_length]
        alpha_t = self.softmax(e_t) # shape: [batch, self.output_time_steps, feature_length]
        return alpha_t
    
    def get_config(self):
        return {'w': self.w_e,
                'u': self.u_e,
                'v': self.v_e,
                'units': self.units,
                'name': self.name}
    
class Encoder_convs(tk.Model):
    def __init__(self, filters=[32, 32, 32, 5], sizes=[3, 3, 3, 1], strides=[1, 1, 1, 1], dilation_rates=[], name="enc_convs"):
        super(Encoder_convs, self).__init__(name=name)
        self.filters = filters
        self.sizes = sizes
        self.strides = strides
        # Layers
        self.convs = []
        if len(dilation_rates) == 0:
            for fs, ks, ss in zip(filters, sizes, strides):
                self.convs.append(tk.layers.Conv1D(fs, ks, ss, padding='valid'))
        else:
            for fs, ks, ss, dr in zip(filters, sizes, strides, dilation_rates):
                if dr >= 2:
                    self.convs.append(tk.layers.Conv1D(fs, ks, ss, dilation_rate=dr, padding='valid'))
                else:
                    self.convs.append(tk.layers.Conv1D(fs, ks, ss, padding='valid'))
#         self.relu = tk.layers.LeakyReLU(alpha=0.2)
        self.relu = tk.layers.ReLU()

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
    def __init__(self, past_filters=[32, 32, 32, 5], past_sizes=[3, 3, 3, 1], past_strides=[1, 1, 1, 1],
                 forward_filters=[32, 32, 32, 5], forward_sizes=[3, 3, 3, 1], forward_strides=[2, 2, 2, 1], name="conv_mlp"):
        super(conv1d_mlp, self).__init__(name=name)
        # sub models and layers
        self.enc_past = Encoder_convs(past_filters, past_sizes, past_strides, name="enc_past")
        self.enc_forward = Encoder_convs(forward_filters, forward_sizes, forward_strides, name="enc_forward")
        self.dense1 = tk.layers.Dense(10)
        self.relu = tk.layers.ReLU()
        self.flatten = tk.layers.Flatten()
        self.concat = tk.layers.Concatenate(axis=-1)

    def call(self, x_past, x_forward, training=False):
        code_past = self.flatten(self.enc_past(x_past))
        code_forward = self.flatten(self.enc_forward(x_forward))
        code = self.concat([code_past, code_forward])
        x = self.dense1(code)
        output = tf.reshape(self.relu(x), (-1, 10, 1))

        return output
    
    def get_config(self):
        return {'encoder_past': self.enc_past,
                'encoder_forward': self.enc_forward,
                'dense': self.dense1}

    
class conv1d_att(tk.Model):
    def __init__(self, past_filters=[32, 32, 32, 1], past_sizes=[3, 3, 3, 1], past_strides=[1, 1, 1, 1],
                 forward_filters=[32, 32, 32, 1], forward_sizes=[3, 3, 3, 1], forward_strides=[2, 2, 2, 1],
                 name="conv_att"):
        super(conv1d_att, self).__init__(name=name)
        # sub models and layers
        self.enc_past = Encoder_convs(past_filters, past_sizes, past_strides, "enc_past")
        self.spatial_att = myAttention(5, "spatial_att")
        self.channel_att = myAttention(5, "channel_att")
        self.enc_forward = Encoder_convs(forward_filters, forward_sizes, forward_strides, "enc_forward")
        self.dense1 = tk.layers.Dense(1)
        self.relu = tk.layers.ReLU()
        self.output_time_steps = 10

    def call(self, x_past, x_forward, training=False):
        x_forward = tf.expand_dims(x_forward, axis=1)
        code_past = self.enc_past(x_past)
        query = tf.transpose(tf.concat([tf.tile(code_past, [1, 1, self.output_time_steps]), pos_enc], axis=1), [0, 2, 1])
        alpha = self.spatial_att(query, x_forward)
        att_feature = tf.expand_dims(alpha, axis=3)*x_forward
        beta = self.channel_att(query, tf.transpose(att_feature, [0, 1, 3, 2]))
        att_feature = tf.expand_dims(beta, axis=2)*att_feature
        code = self.enc_forward(att_feature)
        x = self.dense1(tf.squeeze(code, axis=-1))
        output = self.relu(x)

        return output

    def get_config(self):
        return {'encoder_past': self.enc_past,
                'spatial_att': self.spatial_att,
                'channel_att': self.channel_att,
                'encoder_forward': self.enc_forward,
                'dense': self.dense1}
    
class conv1d_att_w_past(tk.Model):
    def __init__(self, past_filters=[32, 32, 32, 1], past_sizes=[3, 3, 3, 1], past_strides=[1, 1, 1, 1],
                 forward_filters=[32, 32, 32, 1], forward_sizes=[3, 3, 3, 1], forward_strides=[2, 2, 2, 1],
                 forward_dilation_rates=[1, 1, 1, 1], name="conv_att_w_past"):
        super(conv1d_att_w_past, self).__init__(name=name)
        # sub models and layers
        self.enc_past = Encoder_convs(past_filters, past_sizes, past_strides, name="enc_past")
        self.spatial_att = myAttention(5, "spatial_att")
        self.channel_att = myAttention(5, "channel_att")
        self.enc_forward = Encoder_convs(forward_filters, forward_sizes, forward_strides, forward_dilation_rates, "enc_forward")
        self.dense1 = tk.layers.Dense(1)
        self.relu = tk.layers.ReLU()
        self.output_time_steps = 10

    def call(self, x_past, x_forward, pos_enc, training=False):
#         x_forward = tf.expand_dims(x_forward, axis=1)
        x_forward = tf.tile(tf.expand_dims(x_forward, axis=1), [1, 10, 1, 1])
        code_past = self.enc_past(x_past) # shape: [-1, k, 1]
        code_past = tf.transpose(tf.tile(code_past, [1, 1, self.output_time_steps]), [0, 2, 1]) # shape: [-1, self.output_time_steps, k]
        query = code_past
#         query = tf.concat([code_past, pos_enc], axis=-1) # shape: [-1, self.output_time_steps, k+self.output_time_steps]
        alpha = self.spatial_att(query, x_forward) # shape: [-1, self.output_time_steps, forward_length]
        att_feature = tf.expand_dims(alpha, axis=3)*x_forward # shape: [-1, self.output_time_steps, forward_length, forward_channel]
        beta = self.channel_att(query, tf.transpose(att_feature, [0, 1, 3, 2])) # shape: [-1, self.output_time_steps, forward_channel]
        att_feature = tf.expand_dims(beta, axis=2)*att_feature # shape: [-1, self.output_time_steps, forward_length, forward_channel]
        code = tf.squeeze(self.enc_forward(att_feature), axis=-1) # shape: [-1, self.output_time_steps, m]
        code = tf.concat([code_past, code], axis=-1) # shape: [-1, self.output_time_steps, k+m]
        x = self.dense1(code) # shape: [-1, self.output_time_steps, 1]
        output = self.relu(x) # shape: [-1, self.output_time_steps, 1]

        return output
    
    def get_config(self):
        return {'encoder_past': self.enc_past,
                'spatial_att': self.spatial_att,
                'channel_att': self.channel_att,
                'encoder_forward': self.enc_forward,
                'dense': self.dense1}
    
class conv1d_att_emb(tk.Model):
    def __init__(self, past_filters=[32, 32, 32, 1], past_sizes=[3, 3, 3, 1], past_strides=[1, 1, 1, 1],
                 forward_filters=[32, 32, 32, 1], forward_sizes=[3, 3, 3, 1], forward_strides=[2, 2, 2, 1],
                 forward_dilation_rates=[1, 1, 1, 1], name="conv1d_att_emb"):
        super(conv1d_att_emb, self).__init__(name=name)
        # sub models and layers
        self.enc_past = Encoder_convs(past_filters, past_sizes, past_strides, name="enc_past")
        self.spatial_att = myAttention(5, "spatial_att")
        self.channel_att = myAttention(5, "channel_att")
        self.enc_forward = Encoder_convs(forward_filters, forward_sizes, forward_strides, forward_dilation_rates, "enc_forward")
        self.dense1 = tk.layers.Dense(1)
        self.relu = tk.layers.ReLU()
        self.output_time_steps = 10
        self.embed = tk.layers.Embedding(self.output_time_steps, 3, input_length=10)

    def call(self, x_past, x_forward, pos_enc, training=False):
        x_forward = tf.expand_dims(x_forward, axis=1)
        code_past = self.enc_past(x_past) # shape: [-1, k, 1]
        code_past = tf.transpose(tf.tile(code_past, [1, 1, self.output_time_steps]), [0, 2, 1]) # shape: [-1, self.output_time_steps, k]
        pos_enc = self.embed(pos_enc) # shape: [-1, self.output_time_steps, k+3]
        query = tf.concat([code_past, pos_enc], axis=-1) # shape: [-1, self.output_time_steps, k+self.output_time_steps]
        alpha = self.spatial_att(query, x_forward) # shape: [-1, self.output_time_steps, forward_length]
        att_feature = tf.expand_dims(alpha, axis=3)*x_forward # shape: [-1, self.output_time_steps, forward_length, forward_channel]
        beta = self.channel_att(query, tf.transpose(att_feature, [0, 1, 3, 2])) # shape: [-1, self.output_time_steps, forward_channel]
        att_feature = tf.expand_dims(beta, axis=2)*att_feature # shape: [-1, self.output_time_steps, forward_length, forward_channel]
        code = tf.squeeze(self.enc_forward(att_feature), axis=-1) # shape: [-1, self.output_time_steps, m]
        code = tf.concat([code_past, code], axis=-1) # shape: [-1, self.output_time_steps, k+m]
        x = self.dense1(code) # shape: [-1, self.output_time_steps, 1]
        output = self.relu(x) # shape: [-1, self.output_time_steps, 1]

        return output
    
    def get_config(self):
        return {'encoder_past': self.enc_past,
                'spatial_att': self.spatial_att,
                'channel_att': self.channel_att,
                'encoder_forward': self.enc_forward,
                'dense': self.dense1}

    
class conv1d_att_last(tk.Model):
    def __init__(self, past_filters=[32, 32, 32, 1], past_sizes=[3, 3, 3, 1], past_strides=[1, 1, 1, 1],
                 forward_filters=[32, 32, 32, 1], forward_sizes=[3, 3, 3, 1], forward_strides=[2, 2, 2, 1],
                 name="conv_att"):
        super(conv1d_att_last, self).__init__(name=name)
        # sub models and layers
        self.enc_past = Encoder_convs(past_filters, past_sizes, past_strides, name="enc_past")
        self.spatial_att = myAttention(5, "spatial_att")
        self.channel_att = myAttention(5, "channel_att")
        self.enc_forward = Encoder_convs(forward_filters, forward_sizes, forward_strides, name="enc_forward")
        self.dense_query = tk.layers.Dense(10)
        self.dense1 = tk.layers.Dense(1)
        self.relu = tk.layers.ReLU()
        self.output_time_steps = 10
        self.embed = tk.layers.Embedding(self.output_time_steps, 3)

    def call(self, x_past, x_forward, pos_enc, training=False):
        code_past = self.enc_past(x_past) # shape: [-1, k, 1]
        code_past = tf.transpose(tf.tile(code_past, [1, 1, self.output_time_steps]), [0, 2, 1]) # shape: [-1, self.output_time_steps, k]
        pos_enc = self.embed(pos_enc)  # shape: [-1, self.output_time_steps, k+3]
        feature = tf.expand_dims(self.enc_forward(x_forward), axis=1)
        query = tf.concat([code_past, pos_enc], axis=-1) # shape: [-1, self.output_time_steps, k+self.output_time_steps]
        query = self.dense_query(query)
        alpha = self.spatial_att(query, feature) # shape: [-1, self.output_time_steps, forward_length]
        att_feature = tf.expand_dims(alpha, axis=3)*feature # shape: [-1, self.output_time_steps, forward_length, forward_channel]
        beta = self.channel_att(query, tf.transpose(att_feature, [0, 1, 3, 2])) # shape: [-1, self.output_time_steps, forward_channel]
        att_feature = tf.expand_dims(beta, axis=2)*att_feature # shape: [-1, self.output_time_steps, forward_length, forward_channel]
        code = tf.reshape(att_feature, (-1, self.output_time_steps, att_feature.shape[-1]*att_feature.shape[-2])) # shape: [-1, self.output_time_steps, m]
        code = tf.concat([code_past, code], axis=-1) # shape: [-1, self.output_time_steps, k+m]
        x = self.dense1(code) # shape: [-1, self.output_time_steps, 1]
        output = self.relu(x) # shape: [-1, self.output_time_steps, 1]

        return output
    
    def get_config(self):
        return {'encoder_past': self.enc_past,
                'spatial_att': self.spatial_att,
                'channel_att': self.channel_att,
                'encoder_forward': self.enc_forward,
                'dense': self.dense1}

class conv1d_att_pos_enc(tk.Model):
    def __init__(self, past_filters=[32, 32, 32, 1], past_sizes=[3, 3, 3, 1], past_strides=[1, 1, 1, 1],
                 forward_filters=[32, 32, 32, 1], forward_sizes=[3, 3, 3, 1], forward_strides=[2, 2, 2, 1],
                 name="conv_att"):
        super(conv1d_att_pos_enc, self).__init__(name=name)
        # sub models and layers
        self.enc_past = Encoder_convs(past_filters, past_sizes, past_strides, name="enc_past")
        self.spatial_att = myAttention(5, "spatial_att")
        self.channel_att = myAttention(5, "channel_att")
        self.enc_forward = Encoder_convs(forward_filters, forward_sizes, forward_strides, name="enc_forward")
        self.dense1 = tk.layers.Dense(1)
        self.relu = tk.layers.ReLU()
        self.output_time_steps = 10

    def call(self, x_past, x_forward, pos_enc_past, pos_enc_fwd, training=False):
        x_forward = tf.expand_dims(tf.concat([x_forward, pos_enc_fwd], axis=-1), axis=1)
        code_past = self.enc_past(x_past) # shape: [-1, k, 1]
        code_past = tf.transpose(tf.tile(code_past, [1, 1, self.output_time_steps]), [0, 2, 1]) # shape: [-1, self.output_time_steps, k]
        query = tf.concat([code_past, pos_enc_past], axis=-1) # shape: [-1, self.output_time_steps, k+self.output_time_steps]
        alpha = self.spatial_att(query, x_forward) # shape: [-1, self.output_time_steps, forward_length]
        att_feature = tf.expand_dims(alpha, axis=3)*x_forward # shape: [-1, self.output_time_steps, forward_length, forward_channel]
        beta = self.channel_att(query, tf.transpose(att_feature, [0, 1, 3, 2])) # shape: [-1, self.output_time_steps, forward_channel]
        att_feature = tf.expand_dims(beta, axis=2)*att_feature # shape: [-1, self.output_time_steps, forward_length, forward_channel]
        code = tf.squeeze(self.enc_forward(att_feature), axis=-1) # shape: [-1, self.output_time_steps, m]
        code = tf.concat([code_past, code], axis=-1) # shape: [-1, self.output_time_steps, k+m]
        x = self.dense1(code) # shape: [-1, self.output_time_steps, 1]
        output = self.relu(x) # shape: [-1, self.output_time_steps, 1]

        return output

    def get_config(self):
        return {'encoder_past': self.enc_past,
                'spatial_att': self.spatial_att,
                'channel_att': self.channel_att,
                'encoder_forward': self.enc_forward,
                'dense': self.dense1}
    
class conv1d_att_w_pos_fwd(tk.Model):
    def __init__(self, past_filters=[32, 32, 32, 1], past_sizes=[3, 3, 3, 1], past_strides=[1, 1, 1, 1],
                 forward_filters=[32, 32, 32, 1], forward_sizes=[3, 3, 3, 1], forward_strides=[2, 2, 2, 1],
                 name="conv_att"):
        super(conv1d_att_w_pos_fwd, self).__init__(name=name)
        # sub models and layers
        self.enc_past = Encoder_convs(past_filters, past_sizes, past_strides, name="enc_past")
        self.spatial_att = myAttention(16, "spatial_att")
#         self.channel_att = myAttention(16, "channel_att")
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
#         code_past = self.enc_past(x_past) # shape: [-1, k, 1]
        code_past = self.relu(self.dense1(self.flatten(self.enc_past(x_past))))
        code_past = tf.tile(tf.expand_dims(code_past, axis=1), [1, self.output_time_steps, 1])        
        query = self.relu(self.dense2(tf.concat([code_past, pos_enc_past], axis=-1))) # shape: [-1, self.output_time_steps, k+self.output_time_steps]
#         query = self.relu(self.dense2(code_past+pos_enc_past)) # shape: [-1, self.output_time_steps, k+self.output_time_steps]
        alpha = self.spatial_att(query, tf.concat([x_forward, pos_enc_fwd], axis=-1)) # shape: [-1, self.output_time_steps, forward_length]
        att_feature = tf.expand_dims(alpha, axis=3)*x_forward # shape: [-1, self.output_time_steps, forward_length, forward_channel]
#         beta = self.channel_att(query, tf.transpose(att_feature, [0, 1, 3, 2])) # shape: [-1, self.output_time_steps, forward_channel]
#         att_feature = tf.expand_dims(beta, axis=2)*att_feature # shape: [-1, self.output_time_steps, forward_length, forward_channel]
        code = self.relu(self.dense3(tf.reshape(self.enc_forward(att_feature), [-1, self.output_time_steps, 94*32]))) # shape: [-1, self.output_time_steps, m]
        code = tf.concat([query, code], axis=-1) # shape: [-1, self.output_time_steps, k+m]
        x = self.relu(self.dense4(code)) # shape: [-1, self.output_time_steps, 1]
#         output = self.relu(self.dense5(x)) # shape: [-1, self.output_time_steps, 1]
        output = self.tanh(self.dense5(x)) + x_past[:,-1:,-1:]

        return output

    def get_config(self):
        return {'encoder_past': self.enc_past,
                'spatial_att': self.spatial_att,
#                 'channel_att': self.channel_att,
                'encoder_forward': self.enc_forward,
                'dense': self.dense1}


class conv1d_pos_enc(tk.Model):
    def __init__(self, past_filters=[32, 32, 32, 5], past_sizes=[3, 3, 3, 1], past_strides=[1, 1, 1, 1],
                 forward_filters=[32, 32, 32, 5], forward_sizes=[3, 3, 3, 1], forward_strides=[2, 2, 2, 1], name="conv_mlp"):
        super(conv1d_pos_enc, self).__init__(name=name)
        # sub models and layers
        self.enc_past = Encoder_convs(past_filters, past_sizes, past_strides, name="enc_past")
        self.enc_forward = Encoder_convs(forward_filters, forward_sizes, forward_strides, name="enc_forward")
#         self.dense1 = tk.layers.Dense(32)
#         self.dense2 = tk.layers.Dense(32)
#         self.dense3 = tk.layers.Dense(32)
        self.dense4 = tk.layers.Dense(1)
        self.relu = tk.layers.ReLU()
        self.flatten = tk.layers.Flatten()
        self.concat = tk.layers.Concatenate(axis=-1)
        self.output_time_steps = 10

    def call(self, x_past, x_forward, pos_enc_past, training=False):
#         code_past = self.relu(self.dense1(self.flatten(self.enc_past(x_past))))
        code_past = self.flatten(self.enc_past(x_past))
        code_past = tf.tile(tf.expand_dims(code_past, axis=1), [1, self.output_time_steps, 1])
#         code_forward = self.relu(self.dense2(self.flatten(self.enc_forward(x_forward))))
        code_forward = self.flatten(self.enc_forward(x_forward))
        code_forward = tf.tile(tf.expand_dims(code_forward, axis=1), [1, self.output_time_steps, 1])
        code = self.concat([code_past, code_forward]) + pos_enc_past
#         x = self.relu(self.dense3(code))
        x = self.relu(self.dense4(code))

        return x
    
    def get_config(self):
        return {'encoder_past': self.enc_past,
                'encoder_forward': self.enc_forward,
#                 'dense': self.dense1
               }

class conv1d_mlp_2(tk.Model):
    def __init__(self, past_filters=[32, 32, 32, 5], past_sizes=[3, 3, 3, 1], past_strides=[1, 1, 1, 1],
                 forward_filters=[32, 32, 32, 5], forward_sizes=[3, 3, 3, 1], forward_strides=[2, 2, 2, 1], name="conv_mlp"):
        super(conv1d_mlp_2, self).__init__(name=name)
        # sub models and layers
        self.enc_past = Encoder_convs(past_filters, past_sizes, past_strides, name="enc_past")
        self.enc_forward = Encoder_convs(forward_filters, forward_sizes, forward_strides, name="enc_forward")
        self.dense1 = tk.layers.Dense(32)
        self.dense2 = tk.layers.Dense(32)
        self.dense3 = tk.layers.Dense(32)
        self.dense4 = tk.layers.Dense(10)
        self.relu = tk.layers.ReLU()
        self.flatten = tk.layers.Flatten()
        self.concat = tk.layers.Concatenate(axis=-1)

    def call(self, x_past, x_forward, training=False):
        code_past = self.flatten(self.enc_past(x_past))
        code_past = self.relu(self.dense1(code_past))
        code_forward = self.flatten(self.enc_forward(x_forward))
        code_forward = self.relu(self.dense2(code_forward))
        code = self.concat([code_past, code_forward])
        x = self.relu(self.dense3(code))
        x = self.relu(self.dense4(x))
        output = tf.reshape(x, (-1, 10, 1))

        return output
    
    def get_config(self):
        return {'encoder_past': self.enc_past,
                'encoder_forward': self.enc_forward,
                'dense': self.dense1}    


class conv1d_past(tk.Model):
    def __init__(self, name="conv_mlp"):
        super(conv1d_past, self).__init__(name=name)
        # sub models and layers
        self.enc_past = Encoder_origin()
        self.dense1 = tk.layers.Dense(10)
        self.relu = tk.layers.ReLU()
        self.flatten = tk.layers.Flatten()

    def call(self, inputs, training=False):
        x_past = inputs
        code_past = self.flatten(self.enc_past(x_past))
        x = self.dense1(code_past)
        output = tf.reshape(self.relu(x), (-1, 10, 1))

        return output
