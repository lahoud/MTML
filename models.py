
# coding: utf-8

# In[1]:


import numpy as np
import h5py
import os
import random
import pandas as pd
import tensorflow as tf


# In[ ]:


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

## Convolution and Pooling
# Convolution here: stride=1, zero-padded -> output size = input size
def conv3d(x, W):
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME') # conv2d, [1, 1, 1, 1]

def conv3d_stride(x, W, s):
  return tf.nn.conv3d(x, W, strides=[1, s, s, s, 1], padding='SAME') # conv2d, [1, 1, 1, 1]

# Pooling: max pooling over 2x2 blocks
def max_pool_2x2(x):  # tf.nn.max_pool. ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]
  return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


    # neural network loss
    alpha = tf.constant(1,dtype='float32')
    beta = tf.constant(1,dtype='float32')
    gamma = tf.constant(0.001,dtype='float32')#-0.001,dtype='float32')
    delta_d = tf.constant(1.5,dtype='float32')
    delta_v = tf.constant(0.5,dtype='float32')


    labels = tf.reshape(y_,[-1])
    features = tf.reshape(h_conv3,[-1,embed_size])#tf.transpose(tf.reshape(h_conv3,[embed_size,-1]))#
    centers_batch = [tf.reduce_mean(tf.boolean_mask(features,tf.equal(labels,tf.constant(i,dtype='int32'))),axis=0) for i in range(1,nl_inst+1)]
    centers_batch = tf.convert_to_tensor(centers_batch)

    centers_batch = centers_batch + tf.constant(np.random.rand(nl_inst,embed_size)*1e-6,dtype='float32')################ needed when only one voxel is labeled or two groups have the same center to make the difference !=0
    #######################################################################################################
    features_relative = tf.reshape(h_conv3_1,[-1,embed_size])
    centers_batch_relative = [tf.reduce_mean(tf.boolean_mask(features_relative,tf.equal(labels,tf.constant(i,dtype='int32'))),axis=0) for i in range(1,nl_inst+1)]
    centers_batch_relative = tf.convert_to_tensor(centers_batch_relative)
    centers_batch_relative = centers_batch_relative + tf.constant(np.random.rand(nl_inst,embed_size)*1e-6,dtype='float32')

    l_reg = tf.norm(centers_batch_relative, axis=1)
    l_reg = tf.boolean_mask(l_reg,tf.logical_not(tf.is_nan(l_reg)))
    nl_inst_frame = tf.cast(tf.size(l_reg),tf.float32)
    l_reg = tf.reduce_sum(l_reg)
    l_reg = tf.divide(l_reg,nl_inst_frame)

    #######################################################################################################

    valid_voxel_idx = tf.greater(labels,tf.constant(0,dtype='int32'))
    labels2 = tf.boolean_mask(labels,valid_voxel_idx)
    features2 = tf.boolean_mask(features,valid_voxel_idx)

    centers_batch1 = tf.gather(centers_batch, labels2 - tf.constant(1,dtype='int32')) # subtract 1 since the centers skip label 0 (double-check)
    y_l, idx_, count_l = tf.unique_with_counts(labels2)
    to_divide_tens = tf.gather(count_l,idx_)
    loss_1_ = tf.norm(features2 - centers_batch1, axis=1)# before was tf.nn.l2_loss(features - centers_batch1)
    loss_1 = tf.boolean_mask(loss_1_,tf.logical_not(tf.is_nan(loss_1_)))
    to_divide_tens = tf.boolean_mask(to_divide_tens,tf.logical_not(tf.is_nan(loss_1_)))

    # hinge
    where = tf.greater(loss_1, delta_v)
    l_var = tf.multiply(loss_1-delta_v,tf.cast(where,tf.float32)) # try replacing this with boolean mask
    l_var = tf.square(l_var)
    l_var = tf.divide(l_var,tf.cast(to_divide_tens,tf.float32))
    l_var = tf.divide(l_var,nl_inst_frame)
    l_var = tf.boolean_mask(l_var,tf.logical_not(tf.is_nan(l_var)))# can remove this (check no nans reach here)
    l_var = tf.reduce_sum(l_var)

    #######################################################################################################

    l_dist = [tf.subtract(centers_batch[i,:],centers_batch[b,:]) for i in range(1,nl_inst) for b in range(1,nl_inst) if i!=b]
    l_dist = tf.convert_to_tensor(l_dist)
    l_dist = tf.boolean_mask(l_dist,tf.logical_not(tf.is_nan(tf.reduce_sum(l_dist,axis=1))))
    l_dist = tf.norm(l_dist, axis=1)
    where = tf.less(l_dist, 2*delta_d)
    l_dist = tf.multiply(l_dist-2*delta_d,tf.cast(where,tf.float32))
    l_dist = tf.square(l_dist)
    l_dist = tf.divide(l_dist,nl_inst_frame*(nl_inst_frame-1))
    l_dist = tf.boolean_mask(l_dist,tf.logical_not(tf.is_nan(l_dist)))# can remove this (check no nans reach here)
    l_dist = tf.reduce_sum(l_dist)

    #######################################################################################################
    loss = tf.add_n([alpha*l_var,beta*l_dist,gamma*l_reg])
    
    return loss

def get_loss_feat(y_,h_conv3,embed_size,nl_inst):
    # neural network loss
    alpha = tf.constant(1,dtype='float32')
    beta = tf.constant(1,dtype='float32')
    gamma = tf.constant(0.001,dtype='float32')#-0.001,dtype='float32')
    delta_d = tf.constant(1.5,dtype='float32')
    delta_v = tf.constant(0.5,dtype='float32')


    labels = tf.reshape(y_,[-1])
    features = tf.reshape(h_conv3,[-1,embed_size])#tf.transpose(tf.reshape(h_conv3,[embed_size,-1]))#
    centers_batch = [tf.reduce_mean(tf.boolean_mask(features,tf.equal(labels,tf.constant(i,dtype='int32'))),axis=0) for i in range(1,nl_inst+1)]
    centers_batch = tf.convert_to_tensor(centers_batch)

    centers_batch = centers_batch + tf.constant(np.random.rand(nl_inst,embed_size)*1e-6,dtype='float32')################ needed when only one voxel is labeled or two groups have the same center to make the difference !=0
    #######################################################################################################

    l_reg = tf.norm(centers_batch, axis=1)
    l_reg = tf.boolean_mask(l_reg,tf.logical_not(tf.is_nan(l_reg)))
    nl_inst_frame = tf.cast(tf.size(l_reg),tf.float32)
    l_reg = tf.reduce_sum(l_reg)
    l_reg = tf.divide(l_reg,nl_inst_frame)

    #######################################################################################################

    valid_voxel_idx = tf.greater(labels,tf.constant(0,dtype='int32'))
    labels2 = tf.boolean_mask(labels,valid_voxel_idx)
    features2 = tf.boolean_mask(features,valid_voxel_idx)

    centers_batch1 = tf.gather(centers_batch, labels2 - tf.constant(1,dtype='int32')) # subtract 1 since the centers skip label 0 (double-check)
    y_l, idx_, count_l = tf.unique_with_counts(labels2)
    to_divide_tens = tf.gather(count_l,idx_)
    loss_1_ = tf.norm(features2 - centers_batch1, axis=1)# before was tf.nn.l2_loss(features - centers_batch1)
    loss_1 = tf.boolean_mask(loss_1_,tf.logical_not(tf.is_nan(loss_1_)))
    to_divide_tens = tf.boolean_mask(to_divide_tens,tf.logical_not(tf.is_nan(loss_1_)))

    # hinge
    where = tf.greater(loss_1, delta_v)
    l_var = tf.multiply(loss_1-delta_v,tf.cast(where,tf.float32)) # try replacing this with boolean mask
    l_var = tf.square(l_var)
    l_var = tf.divide(l_var,tf.cast(to_divide_tens,tf.float32))
    l_var = tf.divide(l_var,nl_inst_frame)
    l_var = tf.boolean_mask(l_var,tf.logical_not(tf.is_nan(l_var)))# can remove this (check no nans reach here)
    l_var = tf.reduce_sum(l_var)

    #######################################################################################################

    l_dist = [tf.subtract(centers_batch[i,:],centers_batch[b,:]) for i in range(1,nl_inst) for b in range(1,nl_inst) if i!=b]
    l_dist = tf.convert_to_tensor(l_dist)
    l_dist = tf.boolean_mask(l_dist,tf.logical_not(tf.is_nan(tf.reduce_sum(l_dist,axis=1))))
    l_dist = tf.norm(l_dist, axis=1)
    where = tf.less(l_dist, 2*delta_d)
    l_dist = tf.multiply(l_dist-2*delta_d,tf.cast(where,tf.float32))
    l_dist = tf.square(l_dist)
    l_dist = tf.divide(l_dist,nl_inst_frame*(nl_inst_frame-1))
    l_dist = tf.boolean_mask(l_dist,tf.logical_not(tf.is_nan(l_dist)))# can remove this (check no nans reach here)
    l_dist = tf.reduce_sum(l_dist)

    #######################################################################################################
    loss = tf.add_n([alpha*l_var,beta*l_dist,gamma*l_reg])
    
    return loss




def get_loss_physcenter(y_,h_conv3,embed_size,nl_inst,params_size):    
    labels = tf.reshape(y_,[-1])
    features = tf.reshape(h_conv3,[-1,embed_size])
    
    width = params_size[0]
    height = params_size[1]
    depth = params_size[2]
    
    voxel_locations_np = np.transpose(np.indices([width,height,depth]), (1, 2, 3, 0))
    voxel_locations = tf.constant(voxel_locations_np,dtype='float32')    
    voxel_locations = tf.reshape(voxel_locations,[-1,3])
    
    phys_centers_batch = [tf.reduce_mean(tf.boolean_mask(voxel_locations,tf.equal(labels,tf.constant(i,dtype='int32'))),axis=0) for i in range(1,nl_inst+1)]
    phys_centers_batch = tf.convert_to_tensor(phys_centers_batch)
    
    temp_to_count = tf.norm(phys_centers_batch, axis=1)
    temp_to_count = tf.boolean_mask(temp_to_count,tf.logical_not(tf.is_nan(temp_to_count)))
    nl_inst_frame = tf.cast(tf.size(temp_to_count),tf.float32)
    
    
    #
    valid_voxel_idx = tf.greater(labels,tf.constant(0,dtype='int32'))
    labels2 = tf.boolean_mask(labels,valid_voxel_idx)
    features2 = tf.boolean_mask(features,valid_voxel_idx)
    voxel_locations2 = tf.boolean_mask(voxel_locations,valid_voxel_idx)

    vox_phys_center = tf.gather(phys_centers_batch, labels2 - tf.constant(1,dtype='int32')) # subtract 1 since the centers skip label 0 (double-check)
    y_l, idx_, count_l = tf.unique_with_counts(labels2)
    to_divide_tens = tf.gather(count_l,idx_)
    
    # get normalized difference
    vox_to_center = tf.subtract(vox_phys_center,voxel_locations2)
    vox_to_center = tf.nn.l2_normalize(vox_to_center,dim=1)
    
    # normalize features
    features_normalized = tf.nn.l2_normalize(features2,dim=1)# y dim not axis?
    
    # dot product loss
    loss_1_ = tf.multiply(vox_to_center,features_normalized)
    loss_1_ = -tf.reduce_sum(loss_1_,axis=1)
    
    # additional check (might not be needed)
    loss_1 = tf.boolean_mask(loss_1_,tf.logical_not(tf.is_nan(loss_1_)))
    to_divide_tens = tf.boolean_mask(to_divide_tens,tf.logical_not(tf.is_nan(loss_1_)))
    
    #
    l_vec = tf.divide(loss_1,tf.cast(to_divide_tens,tf.float32))
    l_vec = tf.divide(l_vec,nl_inst_frame)
    l_vec = tf.boolean_mask(l_vec,tf.logical_not(tf.is_nan(l_vec)))# can remove this (check no nans reach here)
    loss = tf.reduce_sum(l_vec)
    
    return loss




def sscnet_multi_test_h_pred(x,y_,params_size,test_params,batch_size):
    
    width = params_size[0]
    height = params_size[1]
    depth = params_size[2]
    nLabel = params_size[3]
    embed_size = params_size[4]
    nl_inst = params_size[5]
    
    ta = test_params[0]
    tb = test_params[1]
    tc = test_params[2]
    
    # Input 
    #x_image = tf.cast(tf.expand_dims(tf.argmax(x,axis=4),axis=4),tf.float32)
    #x_image = tf.multiply(x_image,tf.constant(1/nLabel,tf.float32))
   
    # conv1
    W_conv1_1 = weight_variable([7, 7, 7, 40, int(16//ta)])
    b_conv1_1 = bias_variable([int(16//ta)])
    h_conv1_1 = tf.nn.relu(conv3d(x, W_conv1_1) + b_conv1_1)

    W_conv1_2 = weight_variable([3, 3, 3, int(16//ta), int(32//ta)])
    b_conv1_2 = bias_variable([int(32//ta)])
    h_conv1_2 = tf.nn.relu(conv3d(h_conv1_1, W_conv1_2) + b_conv1_2)

    W_conv1_3 = weight_variable([3, 3, 3, int(32//ta), int(32//ta)])
    b_conv1_3 = bias_variable([int(32//ta)])
    h_conv1_3 = conv3d(h_conv1_2, W_conv1_3) + b_conv1_3

    W_conv1_1_t = weight_variable([1, 1, 1, int(16//ta), int(32//ta)])
    b_conv1_1_t = bias_variable([int(32//ta)])
    h_conv1_1_t = conv3d(h_conv1_1, W_conv1_1_t) + b_conv1_1_t

    h_conv1 = tf.add(h_conv1_3,h_conv1_1_t)
    h_pool1 = max_pool_2x2(h_conv1)

    #conv2
    W_conv2_1 = weight_variable([3, 3, 3, int(32//ta), int(64//ta)])
    b_conv2_1 = bias_variable([int(64//ta)])
    h_conv2_1 = tf.nn.relu(conv3d(h_pool1, W_conv2_1) + b_conv2_1)

    W_conv2_2 = weight_variable([3, 3, 3, int(64//ta), int(64//ta)])
    b_conv2_2 = bias_variable([int(64//ta)])
    h_conv2_2 = conv3d(h_conv2_1, W_conv2_2) + b_conv2_2

    W_conv2_1_t = weight_variable([1, 1, 1, int(32//ta), int(64//ta)])
    b_conv2_1_t = bias_variable([int(64//ta)])
    h_conv2_1_t = conv3d(h_pool1, W_conv2_1_t) + b_conv2_1_t 

    h_conv2 = tf.add(h_conv2_2,h_conv2_1_t)
    h_conv2 = tf.nn.relu(h_conv2)

    #conv3
    W_conv3_1 = weight_variable([3, 3, 3, int(64//ta), int(64//ta)])
    b_conv3_1 = bias_variable([int(64//ta)])
    h_conv3_1 = tf.nn.relu(conv3d(h_conv2, W_conv3_1) + b_conv3_1)

    W_conv3_2 = weight_variable([3, 3, 3, int(64//ta), int(64//ta)])
    b_conv3_2 = bias_variable([int(64//ta)])
    h_conv3_2 = conv3d(h_conv3_1, W_conv3_2) + b_conv3_2

    h_conv3 = tf.add(h_conv2,h_conv3_2) 
    h_conv3 = tf.nn.relu(h_conv3)


    #dilate1
    W_dilate1_1 = weight_variable([3, 3, 3, int(64//ta), int(64//ta)])
    b_dilate1_1 = bias_variable([int(64//ta)])
    dilate_1_1 = tf.nn.convolution(h_conv3,W_dilate1_1, strides=[1, 1, 1], padding='SAME', \
                                   dilation_rate=[tb, tb, tb])
    dilate_1_1 = tf.nn.relu(dilate_1_1 + b_dilate1_1)

    W_dilate1_2 = weight_variable([3, 3, 3, int(64//ta), int(64//ta)])
    b_dilate1_2 = bias_variable([int(64//ta)])
    dilate_1_2 = tf.nn.convolution(dilate_1_1,W_dilate1_2, strides=[1, 1, 1], padding='SAME',\
                                   dilation_rate=[tb, tb, tb]) + b_dilate1_2

    h_dilate1 = tf.nn.relu(dilate_1_2)


    #dilate2
    W_dilate2_1 = weight_variable([3, 3, 3, int(64//ta), int(64//ta)])
    b_dilate2_1 = bias_variable([int(64//ta)])
    dilate_2_1 = tf.nn.convolution(h_dilate1,W_dilate2_1, strides=[1, 1, 1], padding='SAME', \
                                   dilation_rate=[tc, tc, tc])
    dilate_2_1 = tf.nn.relu(dilate_2_1 + b_dilate2_1)

    W_dilate2_2 = weight_variable([3, 3, 3, int(64//ta), int(64//ta)])
    b_dilate2_2 = bias_variable([int(64//ta)])
    dilate_2_2 = tf.nn.convolution(dilate_2_1,W_dilate2_2, strides=[1, 1, 1], padding='SAME', \
                                   dilation_rate=[tc, tc, tc]) + b_dilate2_2 

    h_dilate2 = tf.add(h_dilate1,dilate_2_2)

    #concat
    concat = tf.concat([h_dilate2, h_dilate1, h_conv3, h_conv2], axis=4)
    concat = tf.nn.relu(concat)

    #deconv1
    W_deconv1 = weight_variable([4, 4, 4, int(256//ta), int(256//ta)])
    b_deconv1= bias_variable([int(256//ta)])
    concat = tf.nn.conv3d_transpose(concat, W_deconv1, output_shape = [batch_size, width, height, depth,int(256//ta)] ,\
                                    strides=[1, 2, 2, 2, 1], padding='SAME')
    concat = concat + b_deconv1

    #concat_conv
    W_concat_conv1 = weight_variable([1, 1, 1, int(256//ta), int(128//ta)])
    b_concat_conv1 = bias_variable([int(128//ta)])
    h_concat_conv1 = tf.nn.relu(conv3d(concat, W_concat_conv1) + b_concat_conv1)


    W_concat_conv2 = weight_variable([1, 1, 1, int(128//ta), int(128//ta)])
    b_concat_conv2 = bias_variable([int(128//ta)])
    h_concat_conv2 = tf.nn.relu(conv3d(h_concat_conv1, W_concat_conv2) + b_concat_conv2)

    W_concat_conv3 = weight_variable([1, 1, 1, int(128//ta), embed_size])
    b_concat_conv3 = bias_variable([embed_size])
    h_concat_conv = conv3d(h_concat_conv2, W_concat_conv3) + b_concat_conv3
   
    feat_dir = h_concat_conv[...,0:3]
    feat_feat = h_concat_conv[...,3:]
    
    loss_all = [get_loss_physcenter(y_[None,b,...],feat_dir[None,b,...],3,nl_inst,params_size) for b in range(batch_size)]
    loss_all = tf.convert_to_tensor(loss_all)
    loss_dir = tf.reduce_mean(loss_all, name = 'loss_dir')

    loss_all_feat = [get_loss_feat(y_[None,b,...],feat_feat[None,b,...],embed_size-3,nl_inst) for b in range(batch_size)]
    loss_all_feat = tf.convert_to_tensor(loss_all_feat)
    loss_feat = tf.reduce_mean(loss_all_feat, name = 'loss_feat')
    
    features = tf.reshape(h_concat_conv,[batch_size,-1,embed_size],name='features')
    return loss_dir,loss_feat, features

