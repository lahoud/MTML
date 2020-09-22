
# coding: utf-8

# In[ ]:


import numpy as np
import h5py
import os
import random
from scipy import ndimage


# In[ ]:


def get_scene_names(fileloc):
    f = open(fileloc,'r')
    scene_names = f.readlines()
    f.close()
    scene_names = [x.strip() for x in scene_names] 
    return scene_names


# In[ ]:


def get_filenames(hdf5_loc, scene_names, shuffle=False):
    global filenames
    filenames = []
    
    for scene in list(scene_names):
        seg_filename = hdf5_loc+'/'+scene+'_seg.hdf5'
        inst_filename = hdf5_loc+'/'+scene+'_inst.hdf5'
        if os.path.exists(seg_filename) and os.path.exists(inst_filename):
            filenames.append([seg_filename,inst_filename])
          
    if shuffle:
        random.shuffle(filenames) 
        
        
def get_filenames2(hdf5_loc, scene_names, shuffle=False):
    filenames = []
    
    for scene in list(scene_names):
        seg_filename = hdf5_loc+'/'+scene+'_seg.hdf5'
        inst_filename = hdf5_loc+'/'+scene+'_inst.hdf5'
        if os.path.exists(seg_filename) and os.path.exists(inst_filename):
            filenames.append([seg_filename,inst_filename])
          
    if shuffle:
        random.shuffle(filenames) 
        
    return filenames

def get_filenames2_test(hdf5_loc, scene_names, shuffle=False):
    filenames = []
    
    for scene in list(scene_names):
        seg_filename = hdf5_loc+'/'+scene+'_seg.hdf5'
        if os.path.exists(seg_filename):
            filenames.append([seg_filename])
          
    if shuffle:
        random.shuffle(filenames) 
        
    return filenames
        

def get_batch(batch_size,params_size,mirror=False,orient=False):
    global filenames
    
    nx = params_size[0]
    ny = params_size[1]
    nz = params_size[2]
    nl_seg = params_size[3]
    
    len_filenames = len(filenames)
    #if len_filenames < batch_size: get_filenames() 
        
    seg_data = np.zeros([batch_size,nx,ny,nz,nl_seg])
    inst_data = np.zeros([batch_size,nx,ny,nz],dtype='int32')
        
    for i in range(batch_size):
        # read frame 
        seg_filename = filenames[i][0]
        f = h5py.File(seg_filename,'r')
        seg_in = np.array(f.get('voxel_grid'))
        inst_filename = filenames[i][1]
        f = h5py.File(inst_filename,'r')
        inst_out = np.array(f.get('voxel_grid')) + 1 # add 1 so that labels don start with zero, next step is necessary to remove the unknown label
        
        if bool(random.getrandbits(1)) and mirror:
            seg_in = np.flip(seg_in, 0)
            inst_out = np.flip(inst_out, 0)
                   
        if orient:# random rotation
            angle = np.random.randint(0,91)
            seg_in = ndimage.rotate(seg_in,angle,order=0,prefilter=False)
            inst_out = ndimage.rotate(inst_out,angle,order=0,prefilter=False)       
            seg_in = seg_in[:nx,:ny,:nz]# not to get out of given space
            inst_out = inst_out[:nx,:ny,:nz]
        
        # remove instance labels that do not have segmentation labels 
        inst_out[seg_in==0] = 0
        
        # make one hot
        #seg_in_onehot = (np.arange(seg_in.max()) == seg_in[...,None]-1).astype(int) # option1
        seg_in_onehot = (np.arange(seg_in.max()+1) == seg_in[...,None]).astype(int) 
        seg_in_onehot[:,:,:,0] = 0 # option2
        #inst_out_onehot = (np.arange(inst_out.max()) == inst_out[...,None]-1).astype(int)

        # make fixed size
        # append to output
        seg_data[i,:seg_in.shape[0],:seg_in.shape[1],:seg_in.shape[2],:seg_in_onehot.shape[3]] = seg_in_onehot
        #inst_data[i,:inst_out.shape[0],:inst_out.shape[1],:inst_out.shape[2],:inst_out_onehot.shape[3]] = inst_out_onehot
        
        # get different labels for different batchs
        #inst_out[inst_out>0] = inst_out[inst_out>0] + i*nl_inst
        inst_data[i,:inst_out.shape[0],:inst_out.shape[1],:inst_out.shape[2]] = inst_out
        
    filenames = filenames[batch_size:]
    return seg_data,inst_data

def get_batch2(batch_size,params_size,filenames,mirror=False,orient=False):
    
    nx = params_size[0]
    ny = params_size[1]
    nz = params_size[2]
    nl_seg = params_size[3]
    
    len_filenames = len(filenames)
    #if len_filenames < batch_size: get_filenames() 
        
    seg_data = np.zeros([batch_size,nx,ny,nz,nl_seg])
    inst_data = np.zeros([batch_size,nx,ny,nz],dtype='int32')
        
    for i in range(batch_size):
        # read frame 
        seg_filename = filenames[i][0]
        f = h5py.File(seg_filename,'r')
        seg_in = np.array(f.get('voxel_grid'))
        inst_filename = filenames[i][1]
        f = h5py.File(inst_filename,'r')
        inst_out = np.array(f.get('voxel_grid')) # + 1 # add 1 so that labels don start with zero, next step is necessary to remove the unknown label
        
        if bool(random.getrandbits(1)) and mirror:
            seg_in = np.flip(seg_in, 0)
            inst_out = np.flip(inst_out, 0)
                   
        if orient:# random rotation
            angle = np.random.randint(0,360)
            seg_in = ndimage.rotate(seg_in,angle,order=0,prefilter=False)
            inst_out = ndimage.rotate(inst_out,angle,order=0,prefilter=False) 
            
            while np.sum(seg_in[0,:,:]) == 0:  
                seg_in = seg_in[1:,:,:]
                inst_out = inst_out[1:,:,:]
            while np.sum(seg_in[:,0,:]) == 0:  
                seg_in = seg_in[:,1:,:] 
                inst_out = inst_out[:,1:,:]
#             while np.sum(seg_in[0,:,:]) == 0:  seg_in = seg_in[1:,:,:]
#             while np.sum(seg_in[:,0,:]) == 0:  seg_in = seg_in[:,1:,:]
#             while np.sum(inst_out[0,:,:]) == 0:  inst_out = inst_out[1:,:,:]
#             while np.sum(inst_out[:,0,:]) == 0:  inst_out = inst_out[:,1:,:]
                
            seg_in = seg_in[:nx,:ny,:nz]# not to get out of given space
            inst_out = inst_out[:nx,:ny,:nz]
        
        # remove instance labels that do not have segmentation labels 
        inst_out[seg_in==0] = 0
        
        # make one hot
        #seg_in_onehot = (np.arange(seg_in.max()) == seg_in[...,None]-1).astype(int) # option1
        seg_in_onehot = (np.arange(seg_in.max()+1) == seg_in[...,None]).astype(int) 
        seg_in_onehot[:,:,:,0] = 0 # option2
        #inst_out_onehot = (np.arange(inst_out.max()) == inst_out[...,None]-1).astype(int)

        # make fixed size
        # append to output
        seg_data[i,:seg_in.shape[0],:seg_in.shape[1],:seg_in.shape[2],:seg_in_onehot.shape[3]] = seg_in_onehot
        #inst_data[i,:inst_out.shape[0],:inst_out.shape[1],:inst_out.shape[2],:inst_out_onehot.shape[3]] = inst_out_onehot
        
        # get different labels for different batchs
        #inst_out[inst_out>0] = inst_out[inst_out>0] + i*nl_inst
        inst_data[i,:inst_out.shape[0],:inst_out.shape[1],:inst_out.shape[2]] = inst_out
        
    filenames = filenames[batch_size:]
    return seg_data,inst_data,filenames


# In[ ]:


def get_batch_v2(batch_size,params_size,filenames,mirror=False,orient=False,croptosize=False):
    
    nx = params_size[0]
    ny = params_size[1]
    nz = params_size[2]
    nl_seg = params_size[3]
    
    len_filenames = len(filenames)
    #if len_filenames < batch_size: get_filenames() 
        
    seg_data = np.zeros([batch_size,nx,ny,nz,nl_seg])
    inst_data = np.zeros([batch_size,nx,ny,nz],dtype='int32')
        
    for i in range(batch_size):
        # read frame 
        seg_filename = filenames[i][0]
        f = h5py.File(seg_filename,'r')
        seg_in = np.array(f.get('voxel_grid'))
        inst_filename = filenames[i][1]
        f = h5py.File(inst_filename,'r')
        inst_out = np.array(f.get('voxel_grid')) # + 1 # add 1 so that labels don start with zero, next step is necessary to remove the unknown label
        
        if bool(random.getrandbits(1)) and mirror:
            seg_in = np.flip(seg_in, 0)
            inst_out = np.flip(inst_out, 0)
                   
        if orient:# random rotation
            angle = np.random.randint(0,360)
            seg_in = ndimage.rotate(seg_in,angle,order=0,prefilter=False)
            inst_out = ndimage.rotate(inst_out,angle,order=0,prefilter=False) 
            
            while np.sum(seg_in[0,:,:]) == 0:  
                seg_in = seg_in[1:,:,:]
                inst_out = inst_out[1:,:,:]
            while np.sum(seg_in[:,0,:]) == 0:  
                seg_in = seg_in[:,1:,:] 
                inst_out = inst_out[:,1:,:]
#             while np.sum(seg_in[0,:,:]) == 0:  seg_in = seg_in[1:,:,:]
#             while np.sum(seg_in[:,0,:]) == 0:  seg_in = seg_in[:,1:,:]
#             while np.sum(inst_out[0,:,:]) == 0:  inst_out = inst_out[1:,:,:]
#             while np.sum(inst_out[:,0,:]) == 0:  inst_out = inst_out[:,1:,:]
                
            seg_in = seg_in[:nx,:ny,:nz]# not to get out of given space
            inst_out = inst_out[:nx,:ny,:nz]
            
        if croptosize:
            seg_in = seg_in[:nx,:ny,:nz]# not to get out of given space
            inst_out = inst_out[:nx,:ny,:nz]
        
        # remove label 40 and larger
        seg_in[seg_in>=40] = 0
        
        # remove instance labels that do not have segmentation labels 
        inst_out[seg_in==0] = 0
        
        # make one hot
        #seg_in_onehot = (np.arange(seg_in.max()) == seg_in[...,None]-1).astype(int) # option1
        seg_in_onehot = (np.arange(seg_in.max()+1) == seg_in[...,None]).astype(int) 
        seg_in_onehot[:,:,:,0] = 0 # option2
        #inst_out_onehot = (np.arange(inst_out.max()) == inst_out[...,None]-1).astype(int)

        # make fixed size
        # append to output
        seg_data[i,:seg_in.shape[0],:seg_in.shape[1],:seg_in.shape[2],:seg_in_onehot.shape[3]] = seg_in_onehot
        #inst_data[i,:inst_out.shape[0],:inst_out.shape[1],:inst_out.shape[2],:inst_out_onehot.shape[3]] = inst_out_onehot
        
        # get different labels for different batchs
        #inst_out[inst_out>0] = inst_out[inst_out>0] + i*nl_inst
        inst_data[i,:inst_out.shape[0],:inst_out.shape[1],:inst_out.shape[2]] = inst_out
        
    filenames = filenames[batch_size:]
    return seg_data,inst_data,filenames

def get_batch_v2_test(batch_size,params_size,filenames,croptosize=False):
    
    nx = params_size[0]
    ny = params_size[1]
    nz = params_size[2]
    nl_seg = params_size[3]
    
    len_filenames = len(filenames)
    #if len_filenames < batch_size: get_filenames() 
        
    seg_data = np.zeros([batch_size,nx,ny,nz,nl_seg])
    inst_data = np.zeros([batch_size,nx,ny,nz],dtype='int32')
        
    for i in range(batch_size):
        # read frame 
        seg_filename = filenames[i][0]
        f = h5py.File(seg_filename,'r')
        seg_in = np.array(f.get('voxel_grid'))
                    
        if croptosize:
            seg_in = seg_in[:nx,:ny,:nz]# not to get out of given space
            inst_out = inst_out[:nx,:ny,:nz]
        
        # remove label 40 and larger
        seg_in[seg_in>=40] = 0
        
        
        # make one hot
        #seg_in_onehot = (np.arange(seg_in.max()) == seg_in[...,None]-1).astype(int) # option1
        seg_in_onehot = (np.arange(seg_in.max()+1) == seg_in[...,None]).astype(int) 
        seg_in_onehot[:,:,:,0] = 0 # option2
        #inst_out_onehot = (np.arange(inst_out.max()) == inst_out[...,None]-1).astype(int)

        # make fixed size
        # append to output
        seg_data[i,:seg_in.shape[0],:seg_in.shape[1],:seg_in.shape[2],:seg_in_onehot.shape[3]] = seg_in_onehot
        #inst_data[i,:inst_out.shape[0],:inst_out.shape[1],:inst_out.shape[2],:inst_out_onehot.shape[3]] = inst_out_onehot
        
        
    filenames = filenames[batch_size:]
    return seg_data,filenames

def get_batch_v2_select_classes(batch_size,params_size,filenames,mirror=False,orient=False):
    
    nx = params_size[0]
    ny = params_size[1]
    nz = params_size[2]
    nl_seg = params_size[3]
    
    len_filenames = len(filenames)
    #if len_filenames < batch_size: get_filenames() 
        
    seg_data = np.zeros([batch_size,nx,ny,nz,nl_seg])
    inst_data = np.zeros([batch_size,nx,ny,nz],dtype='int32')
        
    for i in range(batch_size):
        # read frame 
        seg_filename = filenames[i][0]
        f = h5py.File(seg_filename,'r')
        seg_in = np.array(f.get('voxel_grid'))
        inst_filename = filenames[i][1]
        f = h5py.File(inst_filename,'r')
        inst_out = np.array(f.get('voxel_grid')) # + 1 # add 1 so that labels don start with zero, next step is necessary to remove the unknown label
        
        if bool(random.getrandbits(1)) and mirror:
            seg_in = np.flip(seg_in, 0)
            inst_out = np.flip(inst_out, 0)
                   
        if orient:# random rotation
            angle = np.random.randint(0,360)
            seg_in = ndimage.rotate(seg_in,angle,order=0,prefilter=False)
            inst_out = ndimage.rotate(inst_out,angle,order=0,prefilter=False) 
            
            while np.sum(seg_in[0,:,:]) == 0:  
                seg_in = seg_in[1:,:,:]
                inst_out = inst_out[1:,:,:]
            while np.sum(seg_in[:,0,:]) == 0:  
                seg_in = seg_in[:,1:,:] 
                inst_out = inst_out[:,1:,:]
#             while np.sum(seg_in[0,:,:]) == 0:  seg_in = seg_in[1:,:,:]
#             while np.sum(seg_in[:,0,:]) == 0:  seg_in = seg_in[:,1:,:]
#             while np.sum(inst_out[0,:,:]) == 0:  inst_out = inst_out[1:,:,:]
#             while np.sum(inst_out[:,0,:]) == 0:  inst_out = inst_out[:,1:,:]
                
            seg_in = seg_in[:nx,:ny,:nz]# not to get out of given space
            inst_out = inst_out[:nx,:ny,:nz]
        
        # remove label 40 and larger
        seg_in[seg_in>=40] = 0
        
        # remove instance labels that do not have segmentation labels 
        inst_out[seg_in==0] = 0
        
        # remove instance labels not compared by benchmark
        INVALID_CLASS_IDS = np.array([1,2,13,15,17,18,19,20,21,22,23,25,26,27,29,30,31,32,35,37,38])
        for nn in list(INVALID_CLASS_IDS):
            inst_out[seg_in==nn] = 0
        # relabel instances continuously    
        old_label_idx = np.unique(inst_out)
        new_label_idx = np.arange(old_label_idx.shape[0]) # can be done more concisely
        for ll in range(1,old_label_idx.shape[0]):
            inst_out[inst_out==old_label_idx[ll]] = new_label_idx[ll]
                        
        
        # make one hot
        #seg_in_onehot = (np.arange(seg_in.max()) == seg_in[...,None]-1).astype(int) # option1
        seg_in_onehot = (np.arange(seg_in.max()+1) == seg_in[...,None]).astype(int) 
        seg_in_onehot[:,:,:,0] = 0 # option2
        #inst_out_onehot = (np.arange(inst_out.max()) == inst_out[...,None]-1).astype(int)

        # make fixed size
        # append to output
        seg_data[i,:seg_in.shape[0],:seg_in.shape[1],:seg_in.shape[2],:seg_in_onehot.shape[3]] = seg_in_onehot
        #inst_data[i,:inst_out.shape[0],:inst_out.shape[1],:inst_out.shape[2],:inst_out_onehot.shape[3]] = inst_out_onehot
        
        # get different labels for different batchs
        #inst_out[inst_out>0] = inst_out[inst_out>0] + i*nl_inst
        inst_data[i,:inst_out.shape[0],:inst_out.shape[1],:inst_out.shape[2]] = inst_out
        
    filenames = filenames[batch_size:]
    return seg_data,inst_data,filenames


def get_batch2_color(batch_size,params_size,filenames,mirror=False,orient=False):
    
    nx = params_size[0]
    ny = params_size[1]
    nz = params_size[2]
    nl_seg = params_size[3]
    
    len_filenames = len(filenames)
    #if len_filenames < batch_size: get_filenames() 
        
    seg_data = np.zeros([batch_size,nx,ny,nz,nl_seg])
    inst_data = np.zeros([batch_size,nx,ny,nz],dtype='int32')
    color_data = np.zeros([batch_size,nx,ny,nz,3],dtype='int32')
        
    for i in range(batch_size):
        # read frame 
        seg_filename = filenames[i][1]
        f = h5py.File(seg_filename,'r')
        seg_in = np.array(f.get('voxel_grid'))
        inst_filename = filenames[i][2]
        f = h5py.File(inst_filename,'r')
        inst_out = np.array(f.get('voxel_grid')) # + 1 # add 1 so that labels don start with zero, next step is necessary to remove the unknown label
        color_filename = filenames[i][0]
        f = h5py.File(color_filename,'r')
        color = np.array(f.get('voxel_grid'))
        
        if bool(random.getrandbits(1)) and mirror:
            seg_in = np.flip(seg_in, 0)
            inst_out = np.flip(inst_out, 0)
            color = np.flip(color, 0)
                   
        if orient:# random rotation
            angle = np.random.randint(0,360)
            seg_in = ndimage.rotate(seg_in,angle,order=0,prefilter=False)
            inst_out = ndimage.rotate(inst_out,angle,order=0,prefilter=False) 
            color = ndimage.rotate(color,angle,order=0,prefilter=False)
            
            while np.sum(seg_in[0,:,:]) == 0:  
                seg_in = seg_in[1:,:,:]
                inst_out = inst_out[1:,:,:]
                color = color[1:,:,:,:]
            while np.sum(seg_in[:,0,:]) == 0:  
                seg_in = seg_in[:,1:,:] 
                inst_out = inst_out[:,1:,:]
                color = color[:,1:,:,:]
#             while np.sum(seg_in[0,:,:]) == 0:  seg_in = seg_in[1:,:,:]
#             while np.sum(seg_in[:,0,:]) == 0:  seg_in = seg_in[:,1:,:]
#             while np.sum(inst_out[0,:,:]) == 0:  inst_out = inst_out[1:,:,:]
#             while np.sum(inst_out[:,0,:]) == 0:  inst_out = inst_out[:,1:,:]
                
            seg_in = seg_in[:nx,:ny,:nz]# not to get out of given space
            inst_out = inst_out[:nx,:ny,:nz]
            color = color[:nx,:ny,:nz,:]
        
        # remove instance labels that do not have segmentation labels 
        inst_out[seg_in==0] = 0
        
        # make one hot
        #seg_in_onehot = (np.arange(seg_in.max()) == seg_in[...,None]-1).astype(int) # option1
        seg_in_onehot = (np.arange(seg_in.max()+1) == seg_in[...,None]).astype(int) 
        seg_in_onehot[:,:,:,0] = 0 # option2
        #inst_out_onehot = (np.arange(inst_out.max()) == inst_out[...,None]-1).astype(int)

        # make fixed size
        # append to output
        seg_data[i,:seg_in.shape[0],:seg_in.shape[1],:seg_in.shape[2],:seg_in_onehot.shape[3]] = seg_in_onehot
        #inst_data[i,:inst_out.shape[0],:inst_out.shape[1],:inst_out.shape[2],:inst_out_onehot.shape[3]] = inst_out_onehot
        
        # get different labels for different batchs
        #inst_out[inst_out>0] = inst_out[inst_out>0] + i*nl_inst
        inst_data[i,:inst_out.shape[0],:inst_out.shape[1],:inst_out.shape[2]] = inst_out
        
        color_data[i,:color.shape[0],:color.shape[1],:color.shape[2]] = color
        
    filenames = filenames[batch_size:]
    return color_data, seg_data,inst_data,filenames

def get_filenames2_color(hdf5_loc, color_loc, scene_names, shuffle=False):
    filenames = []
    
    for scene in list(scene_names):
        seg_filename = hdf5_loc+'/'+scene+'_seg.hdf5'
        inst_filename = hdf5_loc+'/'+scene+'_inst.hdf5'
        color_filename = color_loc+'/'+scene+'_color.hdf5'
        if os.path.exists(seg_filename) and os.path.exists(inst_filename) and os.path.exists(color_filename):
            filenames.append([color_filename,seg_filename,inst_filename])
          
    if shuffle:
        random.shuffle(filenames) 
        
    return filenames

def get_batch_v2_h_pred(batch_size,params_size,filenames,mirror=False,orient=False,croptosize=False,normalize_in=False):
    
    tconv_label_to_nyu40 = [0,1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]
    
    nx = params_size[0]
    ny = params_size[1]
    nz = params_size[2]
    nl_seg = params_size[3]
    
    len_filenames = len(filenames)
    #if len_filenames < batch_size: get_filenames() 
        
    seg_data = np.zeros([batch_size,nx,ny,nz,nl_seg])
    inst_data = np.zeros([batch_size,nx,ny,nz],dtype='int32')
    hpred_data = np.zeros([batch_size,nx,ny,nz,nl_seg])
        
    for i in range(batch_size):
        # read frame 
        seg_filename = filenames[i][0]
        f = h5py.File(seg_filename,'r')
        seg_in = np.array(f.get('voxel_grid'))
        inst_filename = filenames[i][1]
        f = h5py.File(inst_filename,'r')
        inst_out = np.array(f.get('voxel_grid')) # + 1 # add 1 so that labels don start with zero, next step is necessary to remove the unknown label
        
        hpred_filename = filenames[i][1][:-9]+'h_pred.hdf5'
        f = h5py.File(hpred_filename,'r')
        h_pred_compact = np.array(f.get('h_pred_compact'))
        h_pred_idx = np.array(f.get('h_pred_idx'))
        
        h_pred_long = np.zeros([h_pred_compact.shape[0],40])
        for kk in range(20):
            h_pred_long[:,tconv_label_to_nyu40[kk+1]] = h_pred_compact[:,kk]
        #h_pred_long --> nyu labels
        #h_pred_compact --> tangent_conv_labels
        
        h_pred_in = np.zeros([seg_in.shape[0],seg_in.shape[1],seg_in.shape[2],nl_seg])
        for nn,idx in enumerate(h_pred_idx):
            if normalize_in:
                #h_pred_in[idx[0],idx[1],idx[2],:] = h_pred_long[nn]/np.sum(h_pred_long[nn],keepdims=True)
                e_pred_long = np.exp(h_pred_long[nn])-1
                h_pred_in[idx[0],idx[1],idx[2],:] = e_pred_long/np.max(e_pred_long,keepdims=True)
            else:
                e_pred_long = np.exp(h_pred_long[nn])-1
                h_pred_in[idx[0],idx[1],idx[2],:] = h_pred_long[nn]#e_pred_long/np.sum(e_pred_long,keepdims=True)
        
        if bool(random.getrandbits(1)) and mirror:
            seg_in = np.flip(seg_in, 0)
            inst_out = np.flip(inst_out, 0)
            h_pred_in = np.flip(h_pred_in, 0)
                   
        if orient:# random rotation
            angle = np.random.randint(0,360)
            seg_in = ndimage.rotate(seg_in,angle,order=0,prefilter=False)
            inst_out = ndimage.rotate(inst_out,angle,order=0,prefilter=False) 
            h_pred_in = ndimage.rotate(h_pred_in,angle,order=0,prefilter=False)
            
            while np.sum(seg_in[0,:,:]) == 0:  
                seg_in = seg_in[1:,:,:]
                inst_out = inst_out[1:,:,:]
                h_pred_in = h_pred_in[1:,:,:,:]
            while np.sum(seg_in[:,0,:]) == 0:  
                seg_in = seg_in[:,1:,:] 
                inst_out = inst_out[:,1:,:]
                h_pred_in = h_pred_in[:,1:,:,:]
#             while np.sum(seg_in[0,:,:]) == 0:  seg_in = seg_in[1:,:,:]
#             while np.sum(seg_in[:,0,:]) == 0:  seg_in = seg_in[:,1:,:]
#             while np.sum(inst_out[0,:,:]) == 0:  inst_out = inst_out[1:,:,:]
#             while np.sum(inst_out[:,0,:]) == 0:  inst_out = inst_out[:,1:,:]
                
            seg_in = seg_in[:nx,:ny,:nz]# not to get out of given space
            inst_out = inst_out[:nx,:ny,:nz]
            h_pred_in = h_pred_in[:nx,:ny,:nz,:]
            
        if croptosize:
            seg_in = seg_in[:nx,:ny,:nz]# not to get out of given space
            inst_out = inst_out[:nx,:ny,:nz]
            h_pred_in = h_pred_in[:nx,:ny,:nz,:]
            
        
        # remove label 40 and larger
        seg_in[seg_in>=40] = 0
        
        # remove instance labels that do not have segmentation labels 
        inst_out[seg_in==0] = 0
        
        # make one hot
        #seg_in_onehot = (np.arange(seg_in.max()) == seg_in[...,None]-1).astype(int) # option1
        seg_in_onehot = (np.arange(seg_in.max()+1) == seg_in[...,None]).astype(int) 
        seg_in_onehot[:,:,:,0] = 0 # option2
        #inst_out_onehot = (np.arange(inst_out.max()) == inst_out[...,None]-1).astype(int)

        # make fixed size
        # append to output
        seg_data[i,:seg_in.shape[0],:seg_in.shape[1],:seg_in.shape[2],:seg_in_onehot.shape[3]] = seg_in_onehot
        #inst_data[i,:inst_out.shape[0],:inst_out.shape[1],:inst_out.shape[2],:inst_out_onehot.shape[3]] = inst_out_onehot
        
        # get different labels for different batchs
        #inst_out[inst_out>0] = inst_out[inst_out>0] + i*nl_inst
        inst_data[i,:inst_out.shape[0],:inst_out.shape[1],:inst_out.shape[2]] = inst_out
        
        hpred_data[i,:h_pred_in.shape[0],:h_pred_in.shape[1],:h_pred_in.shape[2],:h_pred_in.shape[3]] = h_pred_in
        
    filenames = filenames[batch_size:]
    return seg_data,inst_data,filenames,hpred_data

def get_batch_v2_h_pred20(batch_size,params_size,filenames,mirror=False,orient=False,croptosize=False,normalize_in=False):
    
    #tconv_label_to_nyu40 = [0,1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]
    
    nx = params_size[0]
    ny = params_size[1]
    nz = params_size[2]
    nl_seg = params_size[3]
    
    len_filenames = len(filenames)
    #if len_filenames < batch_size: get_filenames() 
        
    seg_data = np.zeros([batch_size,nx,ny,nz,nl_seg])
    inst_data = np.zeros([batch_size,nx,ny,nz],dtype='int32')
    hpred_data = np.zeros([batch_size,nx,ny,nz,20])
        
    for i in range(batch_size):
        # read frame 
        seg_filename = filenames[i][0]
        f = h5py.File(seg_filename,'r')
        seg_in = np.array(f.get('voxel_grid'))
        inst_filename = filenames[i][1]
        f = h5py.File(inst_filename,'r')
        inst_out = np.array(f.get('voxel_grid')) # + 1 # add 1 so that labels don start with zero, next step is necessary to remove the unknown label
        
        hpred_filename = filenames[i][1][:-9]+'h_pred.hdf5'
        f = h5py.File(hpred_filename,'r')
        h_pred_compact = np.array(f.get('h_pred_compact'))
        h_pred_idx = np.array(f.get('h_pred_idx'))
        
        h_pred_long = h_pred_compact#np.zeros([h_pred_compact.shape[0],40])
#         for kk in range(20):
#             h_pred_long[:,tconv_label_to_nyu40[kk+1]] = h_pred_compact[:,kk]
        #h_pred_long --> nyu labels
        #h_pred_compact --> tangent_conv_labels
        
        h_pred_in = np.zeros([seg_in.shape[0],seg_in.shape[1],seg_in.shape[2],20])
        for nn,idx in enumerate(h_pred_idx):
            if normalize_in:
                #h_pred_in[idx[0],idx[1],idx[2],:] = h_pred_long[nn]/np.sum(h_pred_long[nn],keepdims=True)
                #e_pred_long = np.exp(h_pred_long[nn])-1
                e_pred_long = h_pred_long[nn]#np.exp(h_pred_long[nn])
                tmax = np.max(e_pred_long,keepdims=True)
                tmin = np.min(e_pred_long,keepdims=True)
                h_pred_in[idx[0],idx[1],idx[2],:] = (e_pred_long-tmin)/(tmax-tmin)
            else:
                e_pred_long = np.exp(h_pred_long[nn])#-1
                h_pred_in[idx[0],idx[1],idx[2],:] = e_pred_long#/np.max(e_pred_long,keepdims=True)#e_pred_long/np.sum(e_pred_long,keepdims=True)
        
        if bool(random.getrandbits(1)) and mirror:
            seg_in = np.flip(seg_in, 0)
            inst_out = np.flip(inst_out, 0)
            h_pred_in = np.flip(h_pred_in, 0)
                   
        if orient:# random rotation
            angle = np.random.randint(0,360)
            seg_in = ndimage.rotate(seg_in,angle,order=0,prefilter=False)
            inst_out = ndimage.rotate(inst_out,angle,order=0,prefilter=False) 
            h_pred_in = ndimage.rotate(h_pred_in,angle,order=0,prefilter=False)
            
            while np.sum(seg_in[0,:,:]) == 0:  
                seg_in = seg_in[1:,:,:]
                inst_out = inst_out[1:,:,:]
                h_pred_in = h_pred_in[1:,:,:,:]
            while np.sum(seg_in[:,0,:]) == 0:  
                seg_in = seg_in[:,1:,:] 
                inst_out = inst_out[:,1:,:]
                h_pred_in = h_pred_in[:,1:,:,:]
#             while np.sum(seg_in[0,:,:]) == 0:  seg_in = seg_in[1:,:,:]
#             while np.sum(seg_in[:,0,:]) == 0:  seg_in = seg_in[:,1:,:]
#             while np.sum(inst_out[0,:,:]) == 0:  inst_out = inst_out[1:,:,:]
#             while np.sum(inst_out[:,0,:]) == 0:  inst_out = inst_out[:,1:,:]
                
            seg_in = seg_in[:nx,:ny,:nz]# not to get out of given space
            inst_out = inst_out[:nx,:ny,:nz]
            h_pred_in = h_pred_in[:nx,:ny,:nz,:]
            
        if croptosize:
            seg_in = seg_in[:nx,:ny,:nz]# not to get out of given space
            inst_out = inst_out[:nx,:ny,:nz]
            h_pred_in = h_pred_in[:nx,:ny,:nz,:]
            
        
        # remove label 40 and larger
        seg_in[seg_in>=40] = 0
        
        # remove instance labels that do not have segmentation labels 
        inst_out[seg_in==0] = 0
        
        # make one hot
        #seg_in_onehot = (np.arange(seg_in.max()) == seg_in[...,None]-1).astype(int) # option1
        seg_in_onehot = (np.arange(seg_in.max()+1) == seg_in[...,None]).astype(int) 
        seg_in_onehot[:,:,:,0] = 0 # option2
        #inst_out_onehot = (np.arange(inst_out.max()) == inst_out[...,None]-1).astype(int)

        # make fixed size
        # append to output
        seg_data[i,:seg_in.shape[0],:seg_in.shape[1],:seg_in.shape[2],:seg_in_onehot.shape[3]] = seg_in_onehot
        #inst_data[i,:inst_out.shape[0],:inst_out.shape[1],:inst_out.shape[2],:inst_out_onehot.shape[3]] = inst_out_onehot
        
        # get different labels for different batchs
        #inst_out[inst_out>0] = inst_out[inst_out>0] + i*nl_inst
        inst_data[i,:inst_out.shape[0],:inst_out.shape[1],:inst_out.shape[2]] = inst_out
        
        hpred_data[i,:h_pred_in.shape[0],:h_pred_in.shape[1],:h_pred_in.shape[2],:h_pred_in.shape[3]] = h_pred_in
        
    filenames = filenames[batch_size:]
    return seg_data,inst_data,filenames,hpred_data

    
    nx = params_size[0]
    ny = params_size[1]
    nz = params_size[2]
    nl_seg = params_size[3]
    
    len_filenames = len(filenames)
    #if len_filenames < batch_size: get_filenames() 
        
    seg_data = np.zeros([batch_size,nx,ny,nz,nl_seg])
    inst_data = np.zeros([batch_size,nx,ny,nz])
        
    for i in range(batch_size):
        # read frame 
        seg_filename = filenames[i][0]
        f = h5py.File(seg_filename,'r')
        seg_in = np.array(f.get('voxel_grid'))
        inst_filename = filenames[i][1]
        f = h5py.File(inst_filename,'r')
        inst_out = np.array(f.get('voxel_grid')) # + 1 # add 1 so that labels don start with zero, next step is necessary to remove the unknown label
        
        if bool(random.getrandbits(1)) and mirror:
            seg_in = np.flip(seg_in, 0)
            inst_out = np.flip(inst_out, 0)
                   
        if orient:# random rotation
            angle = np.random.randint(0,360)
            seg_in = ndimage.rotate(seg_in,angle,order=0,prefilter=False)
            inst_out = ndimage.rotate(inst_out,angle,order=0,prefilter=False) 
            
            while np.sum(seg_in[0,:,:]) == 0:  
                seg_in = seg_in[1:,:,:]
                inst_out = inst_out[1:,:,:]
            while np.sum(seg_in[:,0,:]) == 0:  
                seg_in = seg_in[:,1:,:] 
                inst_out = inst_out[:,1:,:]
#             while np.sum(seg_in[0,:,:]) == 0:  seg_in = seg_in[1:,:,:]
#             while np.sum(seg_in[:,0,:]) == 0:  seg_in = seg_in[:,1:,:]
#             while np.sum(inst_out[0,:,:]) == 0:  inst_out = inst_out[1:,:,:]
#             while np.sum(inst_out[:,0,:]) == 0:  inst_out = inst_out[:,1:,:]
                
            seg_in = seg_in[:nx,:ny,:nz]# not to get out of given space
            inst_out = inst_out[:nx,:ny,:nz]
            
        if croptosize:
            seg_in = seg_in[:nx,:ny,:nz]# not to get out of given space
            inst_out = inst_out[:nx,:ny,:nz]
        
        # remove label 40 and larger
        seg_in[seg_in>=40] = 0
        
        # remove instance labels that do not have segmentation labels 
        inst_out[seg_in==0] = 0
        
        # make one hot
        #seg_in_onehot = (np.arange(seg_in.max()) == seg_in[...,None]-1).astype(int) # option1
        seg_in_onehot = (np.arange(seg_in.max()+1) == seg_in[...,None]).astype(int) 
        seg_in_onehot[:,:,:,0] = 0 # option2
        #inst_out_onehot = (np.arange(inst_out.max()) == inst_out[...,None]-1).astype(int)

        # make fixed size
        # append to output
        seg_data[i,:seg_in.shape[0],:seg_in.shape[1],:seg_in.shape[2],:seg_in_onehot.shape[3]] = seg_in_onehot
        #inst_data[i,:inst_out.shape[0],:inst_out.shape[1],:inst_out.shape[2],:inst_out_onehot.shape[3]] = inst_out_onehot
        
        # get different labels for different batchs
        #inst_out[inst_out>0] = inst_out[inst_out>0] + i*nl_inst
        inst_data[i,:inst_out.shape[0],:inst_out.shape[1],:inst_out.shape[2]] = inst_out
        
    filenames = filenames[batch_size:]
    return seg_data,inst_data,filenames