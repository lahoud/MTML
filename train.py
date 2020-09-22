
# coding: utf-8

import numpy as np
import argparse
import h5py
import os
import random
import tensorflow as tf
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
import my_toolbox.utils as tools
import models


def main():
    parser = argparse.ArgumentParser(description='Training Script.')
    parser.add_argument('--nx', type=int, default=92, help='Max number of voxels in x direction')
    parser.add_argument('--ny', type=int, default=104, help='Max number of voxels in y direction')
    parser.add_argument('--nz', type=int, default=40, help='Max number of voxels in z direction')    
    parser.add_argument('--nl_seg', type=int, default=40, help='Number of semantic labels')
    parser.add_argument('--nl_inst', type=int, default=50, help='Max number of instances')    
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')     
    parser.add_argument('--embed_size', type=int, default=7, help='Feature embedding size') 
    parser.add_argument('--filter_reduce_rate', type=int, default=2, help='downsize network parameter')
    parser.add_argument('--dilate_rate1', type=int, default=5, help='Network dilation rate 1')
    parser.add_argument('--dilate_rate2', type=int, default=5, help='Network dilation rate 2')    
    parser.add_argument('--alpha_reg', type=float, default=0.5, help='Loss regularizer')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')

    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')  
    parser.add_argument('--checkpoint_loc', type=str, 
                        default='./trained_models/iccv_v2_sscnet_multi_spc_v2_train_hpred_trainval/iter_1000/', 
                        help='Checkpoint location') 

    parser.add_argument('--data_loc', type=str, 
                        default='data/trainval/', 
                        help='Training data location')
    parser.add_argument('--data_loc_val', type=str, 
                        default='data/val/', 
                        help='Validation data location')                         

    parser.add_argument('--max_iter', type=int, default=200000, help='Maximum number of iterations')  
    parser.add_argument('--log_rate', type=int, default=50, help='Logging rate')
    parser.add_argument('--val_rate', type=int, default=100, help='Validation rate')
    parser.add_argument('--val_size', type=int, default=100, help='Number of validation scenes')
    parser.add_argument('--save_rate', type=int, default=500, help='Save rate')
    parser.add_argument('--savedir', type=str, 
                        default="./trained_models/iccv_v2_sscnet_multi_spc_v2_train_hpred_trainval/iter_", 
                        help='Checkpoint save location')    
    args = parser.parse_args()
                     
    ####

    scene_names = tools.get_scene_names('scannet/scannetv2_trainval.txt')
    scene_names_validation = tools.get_scene_names('scannet/scannetv2_val.txt')


    # neural network architecture
    width = args.nx # use the biggest scene size as fixed-size input
    height = args.ny
    depth = args.nz
    nLabel = args.nl_seg

    params_size = np.array([width,height,depth,nLabel,args.embed_size,args.nl_inst])
    test_params = np.array([args.filter_reduce_rate,args.dilate_rate1,args.dilate_rate2])

    ##
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, width,height,depth,nLabel], name='x') # 
    y_ = tf.placeholder(tf.int32, shape=[None, width,height,depth], name='y_')  # 

    loss_dir,loss_dist,features = models.sscnet_multi_test_h_pred(x,y_,params_size,test_params,args.batch_size)

    alpha = tf.constant(args.alpha_reg,dtype='float32')
    loss = loss_dir + alpha*loss_dist

    train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)
    saver = tf.train.Saver()


    ##
    if args.resume:
        saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_loc))
    else:
        sess.run(tf.global_variables_initializer())


    filenames = tools.get_filenames2(args.data_loc, scene_names, shuffle=True)

    # train neural network
    all_losses_dir = np.array(None)
    all_losses_dist = np.array(None)
    for i in range(args.max_iter):
        batchx,batchy,filenames = tools.get_batch_v2(args.batch_size,[args.nx,args.ny,args.nz,args.nl_seg],filenames,mirror=True,orient=True)
        train_step.run(feed_dict={x: batchx, y_: batchy})
    
        # Logging
        if i%args.log_rate == 0:
            train_loss_dir = loss_dir.eval(feed_dict={x:batchx, y_: batchy})
            train_loss_dist = loss_dist.eval(feed_dict={x:batchx, y_: batchy})
            print("step %d, training loss dir: %g ... dist: %g"%(i, train_loss_dir,train_loss_dist))
            all_losses_dir = np.append(all_losses_dir,train_loss_dir)
            all_losses_dist = np.append(all_losses_dist,train_loss_dist)
        
        # validation
        if i%args.val_rate == 0 and i>0:
            filenames = tools.get_filenames2(args.data_loc_val, scene_names_validation)
            losses_validation_dir = np.array(None)
            losses_validation_dist = np.array(None)
            for v in range(args.val_size):
                batchx,batchy,filenames = tools.get_batch_v2(args.batch_size,[args.nx,args.ny,args.nz,args.nl_seg],filenames)
                val_loss_dir = loss_dir.eval(feed_dict={x:batchx, y_: batchy})
                val_loss_dist = loss_dist.eval(feed_dict={x:batchx, y_: batchy})
                losses_validation_dir = np.append(losses_validation_dir,val_loss_dir)
                losses_validation_dist = np.append(losses_validation_dist,val_loss_dist)
            print("Validation loss dir: %g ... dist: %g"%(losses_validation_dir[1:-1].mean(),losses_validation_dist[1:-1].mean()))
            filenames = tools.get_filenames2(args.data_loc, scene_names, shuffle=True)
        
        if i%args.save_rate == 0 and i>0:
            save_path = args.savedir + str(i)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = saver.save(sess, save_path+"/model.ckpt")
            print("Model saved in path: %s" % save_path)


# ################################
