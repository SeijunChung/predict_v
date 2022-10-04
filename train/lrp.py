import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.compat.v1 as tf1
from tensorflow.python.ops import nn_ops
import numpy as np


def get_act_encoder(encoder, x, collect_name='conv'):
    for i in range(len(encoder.convs)):
        x = encoder.convs[i](x)
        x = encoder.relu(x)
        tf1.add_to_collection(collect_name, x)
    return x


def get_act_attention(attention, query, feature, collect_name='att'):
    key = attention.relu(attention.dense1(feature))
    tf1.add_to_collection(collect_name+'_key', key)
    key = attention.relu(attention.dense2(key))
    tf1.add_to_collection(collect_name+'_key', key)
    
    v1 = attention.u_e(key)
    v2 = tf.expand_dims(attention.w_e(query), axis=2)
    tf1.add_to_collection(collect_name+'_v1', v1)
    tf1.add_to_collection(collect_name+'_v2', v2)
    
    e_t = tf.squeeze(attention.v_e(attention.tanh(v1 + v2)), axis=-1)
    tf1.add_to_collection(collect_name+'_e', e_t)
    alpha_t = attention.softmax(e_t)
    
    return alpha_t
    

def get_act_model(model, x_past, x_forward, pos_enc_past, pos_enc_fwd, training=False):
    x_forward = tf.expand_dims(x_forward, axis=1)
    pos_enc_fwd = tf.expand_dims(pos_enc_fwd, axis=1)
    
    past_enc = get_act_encoder(model.enc_past, x_past, collect_name='past_conv')

    code_past = model.relu(model.dense1(model.flatten(past_enc)))
    code_past = tf.tile(tf.expand_dims(code_past, axis=1), [1, model.output_time_steps, 1])
    tf1.add_to_collection('code_past_enc', tf.concat([code_past, pos_enc_past], axis=-1))

    query = model.relu(model.dense2(tf.concat([code_past, pos_enc_past], axis=-1)))  # shape: [-1, self.output_time_steps, k+self.output_time_steps]
    tf1.add_to_collection('query', query)
    
    alpha = get_act_attention(model.spatial_att, query, tf.concat([x_forward, pos_enc_fwd], axis=-1))  # shape: [-1, self.output_time_steps, forward_length]
    tf1.add_to_collection('alpha', alpha)
    
    att_feature = tf.expand_dims(alpha, axis=3) * x_forward  # shape: [-1, self.output_time_steps, forward_length, forward_channel]
    tf1.add_to_collection('att_feature', att_feature)
    forward_enc = get_act_encoder(model.enc_forward, att_feature, collect_name='forward_conv')
    code_att = model.relu(model.dense3(tf.reshape(forward_enc, [-1, model.output_time_steps, 94 * 32])))  # shape: [-1, self.output_time_steps, m]
    
    code = tf.concat([query, code_att], axis=-1)  # shape: [-1, self.output_time_steps, k+m]
    tf1.add_to_collection('concat_code', code)
    
    x = model.relu(model.dense4(code))  # shape: [-1, self.output_time_steps, 1]
    tf1.add_to_collection('dense4', x)
    
    output = model.tanh(model.dense5(x)) + x_past[:, -1:, -1:]
    tf1.add_to_collection('diff', model.tanh(model.dense5(x)))
    
    return output


class RelevancePropagation(object):
    
    def __init__(self):
        self.epsilon = 1e-9
        
    def rp_dense_out(self, x, w, r):
        
        r_p = self.rp_dense(x, w, tf.maximum(r, 0))
        r_n = self.rp_dense(x, -w, tf.abs(tf.minimum(r, 0)))
        
        return r_p + r_n  
    
    def rp_dense_in(self, x, w, r):
        w_2 = tf.pow(w, 2)
        sum_w_2 = tf.reduce_sum(w_2, axis=0)
        return tf.matmul(r, tf.transpose(w_2/sum_w_2))
        
    def rp_dense_neg_parallel(self, x, w, r):
        rel_list = []
        for i in range(10):
            for j in range(100):
                tmp_x = x[i:i+1,j,:]
                tmp_w = tf.transpose(tf.sign(tmp_x))*w
                tmp_x = tmp_x * tf.sign(tmp_x)
                tmp_rel = self.rp_dense(tmp_x, tmp_w, r[i,j])
                rel_list.append(tmp_rel)
                
        return tf.reshape(tf.concat(rel_list, axis=0), (10,100,-1))
        
        
    def rp_dense(self, x, w, r):
        w_pos = tf.maximum(w, 0)
        z = tf.matmul(x, w_pos)
        z = z + self.epsilon
        s = r / z
        c = tf.matmul(s, tf.transpose(w_pos))
        return c * x
    
    def rp_conv(self, x, kernel, r, strides=1, padding='VALID'):
        w_p = tf.maximum(0., kernel)
        z = nn_ops.conv1d(x, w_p, strides, padding) + 1e-10
        s = r / z
        c = tf.nn.conv1d_transpose(s, w_p, output_shape=tf.shape(x), strides=strides, padding=padding)
        return x * c
    
    def rp_conv_parallel(self, x, kernel, r, strides=1, padding='VALID'):
        w_p = tf.maximum(0., kernel)
        rel_list = []
        for i in range(x.shape[1]):
            x2 = x[:,i]
            r2 = r[:,i]
            z = nn_ops.conv1d(x2, w_p, strides, padding) + 1e-10
            s = r2 / z
            c = tf.nn.conv1d_transpose(s, w_p, output_shape=tf.shape(x2), strides=strides, padding=padding)
            rel_list.append( x2 * c )
        return tf.concat(rel_list, axis=0)

    def rp_conv_input(self, x, kernel, r, strides=1, padding='VALID', lowest=-1., highest=1.):
        w_p = tf.maximum(0., kernel)
        w_n = tf.minimum(0., kernel)
        
        L = tf.ones_like(x, tf.float32) * lowest
        H = tf.ones_like(x, tf.float32) * highest
        
        z_o = nn_ops.conv1d(x, kernel, strides, padding)
        z_p = nn_ops.conv1d(L, w_p, strides, padding)
        z_n = nn_ops.conv1d(H, w_n, strides, padding)
        
        z = z_o - z_p - z_n + 1e-10
        s = r / z
        
        c_o =  tf.nn.conv1d_transpose(s, kernel, output_shape=tf.shape(x), strides=strides, padding=padding)
        c_p =  tf.nn.conv1d_transpose(s, w_p, output_shape=tf.shape(x), strides=strides, padding=padding)
        c_n =  tf.nn.conv1d_transpose(s, w_n, output_shape=tf.shape(x), strides=strides, padding=padding)
        
        return x * c_o - L * c_p - H * c_n
    
    def rp_conv_input_parallel(self, x, kernel, r, strides=1, padding='VALID', lowest=-1., highest=1.):
        w_p = tf.maximum(0., kernel)
        w_n = tf.minimum(0., kernel)
        
        rel_list = []
        for i in range(x.shape[1]):
            x2 = x[:,i]
            r2 = r[:,i]
        
            L = tf.ones_like(x2, tf.float32) * lowest
            H = tf.ones_like(x2, tf.float32) * highest

            z_o = nn_ops.conv1d(x2, kernel, strides, padding)
            z_p = nn_ops.conv1d(L, w_p, strides, padding)
            z_n = nn_ops.conv1d(H, w_n, strides, padding)

            z = z_o - z_p - z_n + 1e-10
            s = r2 / z

            c_o =  tf.nn.conv1d_transpose(s, kernel, output_shape=tf.shape(x2), strides=strides, padding=padding)
            c_p =  tf.nn.conv1d_transpose(s, w_p, output_shape=tf.shape(x2), strides=strides, padding=padding)
            c_n =  tf.nn.conv1d_transpose(s, w_n, output_shape=tf.shape(x2), strides=strides, padding=padding)

            rel_list.append(x2 * c_o - L * c_p - H * c_n)
        return tf.concat(rel_list, axis=0)

    def run_lrp_input(self, model,  x_past, x_forward, pos_enc_past, pos_enc_fwd, prnt=False):    
        '''output to concat'''
        tmp_x = tf1.get_collection('dense4')[0]
        tmp_w = model.dense5.get_weights()[0]
        tmp_r = tf1.get_collection('diff')[0]
        rel = self.rp_dense_out(tmp_x, tmp_w, tmp_r)
        
        tmp_x = tf1.get_collection('concat_code')[0]
        tmp_w = model.dense4.get_weights()[0]
        rel2 = self.rp_dense(tmp_x, tmp_w, rel)
        
        '''concat to att_out'''
        tmp_x = tf1.get_collection('forward_conv')[-1]
        tmp_w = model.dense3.get_weights()[0]
        rel3a = self.rp_dense(tf.reshape(tmp_x, (10,-1)) , tmp_w, rel2[:,:,32:])
        
        conv_x = tf1.get_collection('forward_conv')[-2]
        conv_w = model.enc_forward.convs[-1].get_weights()[0]
        rel4a = self.rp_conv_parallel(conv_x, conv_w, tf.reshape(tf.expand_dims(rel3a, axis=0), (-1, model.output_time_steps, 94, 32)))
        
        conv_x = tf1.get_collection('forward_conv')[-3]
        conv_w = model.enc_forward.convs[-2].get_weights()[0]
        rel5a = self.rp_conv_parallel(conv_x, conv_w, tf.reshape(tf.expand_dims(rel4a, axis=0), (-1, model.output_time_steps, 96, 32)))
            
        conv_x = tf1.get_collection('att_feature')[0]
        conv_w = model.enc_forward.convs[-3].get_weights()[0]
        rel6a = self.rp_conv_input_parallel(conv_x, conv_w, tf.reshape(tf.expand_dims(rel5a, axis=0), (-1, model.output_time_steps, 98, 32)))
        
        '''att_out to input'''
        # alpha * x_forward
        alpha = tf1.get_collection('alpha')[0]
        attention = model.spatial_att
        # softmax
        v1 = tf1.get_collection('att_v1')[0]
        v2 = tf1.get_collection('att_v2')[0]
        tmp_x = attention.tanh(v1 + v2)[0]
        tmp_w = attention.v_e.get_weights()[0]
       
        rel6a = tf.expand_dims(tf.reduce_sum(rel6a, axis=2), axis=2)
        rel7a = self.rp_dense_neg_parallel(tmp_x, tmp_w, rel6a)
        
        v1 = tf.tile(v1, [1,10,1,1])
        v2 = tf.tile(v2, [1,1,100,1])
        
        rel_key = rel7a * v1/(v1+v2)
        rel_key = tf.reduce_sum(rel_key, axis=1)
        rel_query = rel7a * v2/(v1+v2)
        rel_query = tf.reduce_sum(rel_query, axis=2)
    
        
        '''att to key'''
        tmp_x = tf1.get_collection('att_key')[-1]
        tmp_w = attention.u_e.get_weights()[0]
        rel_key2 = self.rp_dense_out(tmp_x, tmp_w, rel_key)
        
        tmp_x = tf1.get_collection('att_key')[-2]
        tmp_w = attention.dense2.get_weights()[0]
        rel_key3 = self.rp_dense(tmp_x, tmp_w, rel_key2)
        
        tmp_x = tf.expand_dims(x_forward, axis=1)
        tmp_pos_enc_fwd = tf.expand_dims(pos_enc_fwd, axis=1)
        tmp_x = tf.concat([tmp_x, tmp_pos_enc_fwd], axis=-1)
        tmp_w = attention.dense1.get_weights()[0]
        rel_forward = self.rp_dense_in(tmp_x, tmp_w, rel_key3)
        
        '''att to query''' 
        tmp_x = tf1.get_collection('query')[0]
        tmp_w = attention.w_e.get_weights()[0]
        rel_query2 = self.rp_dense(tmp_x, tmp_w, rel_query)
        
        '''concat to code'''
        tmp_x = tf1.get_collection('code_past_enc')[0]
        tmp_w = model.dense2.get_weights()[0]
        rel3 = self.rp_dense(tmp_x, tmp_w, rel2[:,:,:32]+rel_query2)
        rel_code = tf.reduce_sum(rel3[:], axis=1)
        
        '''code to input'''
        tmp_x = tf1.get_collection('past_conv')[-1]
        tmp_w = model.dense1.get_weights()[0]
        rel4 = self.rp_dense(tf.reshape(tmp_x, (1,-1)), tmp_w, rel_code[:,:32])
        rel4 = tf.reshape(rel4, (1, 14, 32))

        conv_x = tf1.get_collection('past_conv')[-2]
        conv_w = model.enc_past.convs[-1].get_weights()[0]
        rel5 = self.rp_conv(conv_x, conv_w, rel4)
        
        conv_x = tf1.get_collection('past_conv')[-3]
        conv_w = model.enc_past.convs[-2].get_weights()[0]
        rel6 = self.rp_conv(conv_x, conv_w, rel5)
        
        conv_x = x_past
        conv_w = model.enc_past.convs[-3].get_weights()[0]
        rel_input = self.rp_conv_input(conv_x, conv_w, rel6)
        
        past_rel = tf.reduce_sum(rel_input)
        forward_rel = tf.reduce_sum(rel_forward)
        
        if prnt:
            print("[Value]\nPast Relevance: %.6f, Forward Relevance: %.6f"%(past_rel, forward_rel))
            print("[Percentage]\nPast Relevance: %.2f, Forward Relevance: %.2f"%(100*past_rel/(past_rel+forward_rel), 100*forward_rel/(past_rel+forward_rel)))

        return rel_input.numpy(), np.squeeze(rel_forward.numpy(), 0)