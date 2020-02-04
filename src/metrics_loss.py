"""Functions for metric learning for the face recognition network.
"""
# MIT License
#### copyright at Auther: mingzuheng, 25/03/2019 #########


# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf






  
def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)


    """
     #nrof_features = features.get_shape()[1]
     #centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
     #    initializer=tf.constant_initializer(0), trainable=False)
     #label = tf.reshape(label, [-1])
     #centers_batch = tf.gather(centers, label)
     #diff = (1 - alfa) * (centers_batch - features)
     #diff = alfa * (centers_batch - features)
     #centers = tf.scatter_sub(centers, label, diff)
    # loss = tf.nn.l2_loss(features - centers_batch)
    # return loss, centers, diff, centers_batch

    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
       -- mzh 15/02/2017
       -- Correcting the center updating, center updates/shifts towards to the center of the correponding class with a weight:
       -- centers = centers- (1-alpha)(centers-sum(Xi)/Nj), where Xi is the elements of the class j, Nj is the number of the elements of class Nj
       -- code has been tested by the test script '../test/center_loss_test.py'
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
    centers_cts = tf.get_variable('centers_cts', [nrof_classes], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
    #centers_cts_init = tf.zeros_like(nrof_classes, tf.float32)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label) #get the corresponding center of each element in features, the list of the centers is in the same order as the features
    loss_n = tf.reduce_sum(tf.square(features - centers_batch)/2, 1)
    loss = tf.nn.l2_loss(features - centers_batch)
    diff = (1 - alfa) * (centers_batch - features)

    ## update the centers
    label_unique, idx = tf.unique(label)
    zeros = tf.zeros_like(label_unique, tf.float32)
    ## calculation the repeat time of same label
    nrof_elements_per_class_clean = tf.scatter_update(centers_cts, label_unique, zeros)
    ones = tf.ones_like(label, tf.float32)
    ## counting the number elments in each class, the class is in the order of the [0,1,2,3,....] as initialzation
    nrof_elements_per_class_update = tf.scatter_add(nrof_elements_per_class_clean, label, ones)
    ## nrof_elements_per_class_list is the number of the elements in each class in the batch
    nrof_elements_per_class_batch = tf.gather(nrof_elements_per_class_update, label)
    nrof_elements_per_class_batch_reshape = tf.reshape(nrof_elements_per_class_batch, [-1, 1])## reshape the matrix as 1 coloum no matter the dimension of the row (-1)
    diff_mean = tf.div(diff, nrof_elements_per_class_batch_reshape)
    centers = tf.scatter_sub(centers, label, diff_mean)

    #return loss, centers, label, centers_batch, diff, centers_cts, centers_cts_batch, diff_mean,center_cts_clear, nrof_elements_per_class_batch_reshape
    return loss, loss_n, centers, nrof_elements_per_class_clean, nrof_elements_per_class_batch_reshape,diff_mean # facenet_expression_addcnns_simple_joint_v4_dynamic.py
    #return loss, centers, nrof_elements_per_class_clean, nrof_elements_per_class_batch_reshape,diff_mean ### facenet_train_classifier_expression_pretrainExpr_multidata_addcnns_simple.py

def center_loss_similarity(features, label, alfa, nrof_classes):
    ## center_loss on cosine distance =1 - similarity instead of the L2 norm, i.e. Euclidian distance

    ## normalisation as the embedding vectors in order to similarity distance
    features = tf.nn.l2_normalize(features, 1, 1e-10, name='feat_emb')

    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
    centers_cts = tf.get_variable('centers_cts', [nrof_classes], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
    #centers_cts_init = tf.zeros_like(nrof_classes, tf.float32)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label) #get the corresponding center of each element in features, the list of the centers is in the same order as the features
    #loss = tf.nn.l2_loss(features - centers_batch) ## 0.5*(L2 norm)**2, L2 norm is the Euclidian distance
    similarity_all = tf.matmul(features, tf.transpose(tf.nn.l2_normalize(centers_batch, 1, 1e-10))) ## dot prodoct, cosine distance, similarity of x and y
    similarity_self = tf.diag_part(similarity_all)
    loss_x = tf.subtract(1.0, similarity_self)
    loss = tf.reduce_sum(loss_x) ## sum the cosine distance of each vector/tensor
    diff = (1 - alfa) * (centers_batch - features)
    ones = tf.ones_like(label, tf.float32)
    centers_cts = tf.scatter_add(centers_cts, label, ones) # counting the number of each class, the class is in the order of the [0,1,2,3,....] as initialzation
    centers_cts_batch = tf.gather(centers_cts, label)
    #centers_cts_batch_ext = tf.tile(centers_cts_batch, nrof_features)
    #centers_cts_batch_reshape = tf.reshape(centers_cts_batch_ext,[-1, nrof_features])
    centers_cts_batch_reshape = tf.reshape(centers_cts_batch, [-1,1])
    diff_mean = tf.div(diff, centers_cts_batch_reshape)
    centers = tf.scatter_sub(centers, label, diff_mean)
    zeros = tf.zeros_like(label, tf.float32)
    center_cts_clear = tf.scatter_update(centers_cts, label, zeros)
    #return loss, centers, label, centers_batch, diff, centers_cts, centers_cts_batch, diff_mean,center_cts_clear, centers_cts_batch_reshape
    #return loss, centers, loss_x, similarity_all, similarity_self
    return loss, centers




def center_inter_loss_tf(features, nrof_features, label, alfa, nrof_classes): # tensorflow version
    """ center_inter_loss = center_loss/||Xi - centers(0,1,2,...i-1,i+1,i+2,...)||
        --mzh 22022017
    """
    # dim_features = features.get_shape()[1]
    # nrof_features = features.get_shape()[0]
    dim_features = features.get_shape()[1].value
    #nrof_features = features.get_shape()[0].value
    # dim_features = features.shape[1]
    # nrof_features = features.shape[0]
    centers = tf.get_variable('centers', [nrof_classes, dim_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    centers_cts = tf.get_variable('centers_cts', [nrof_classes], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0), trainable=False)
    ## center_loss calculation
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers,label)  # get the corresponding center of each element in features, the list of the centers is in the same order as the features
    dist_centers = features - centers_batch
    dist_centers_sum = tf.reduce_sum(dist_centers**2,1)/2
    loss_center = tf.nn.l2_loss(dist_centers)

    ## calculation the repeat time of same label
    ones = tf.ones_like(label, tf.float32)
    centers_cts = tf.scatter_add(centers_cts, label, ones)  # counting the number of each class, the class is in the order of the [0,1,2,3,....] as initialzation
    centers_cts_batch = tf.gather(centers_cts, label)


    ## inter_center_loss calculation
    #label_unique, label_idx = tf.unique(label)
    #centers_batch1 = tf.gather(centers,label_unique)
    #nrof_classes_batch = centers_batch.get_shape()[0].value
    #centers_1D = tf.reshape(centers_batch1, [1, nrof_classes_batch * dim_features])
    centers_batch1 = tf.gather(centers,label)
    centers_1D = tf.reshape(centers_batch1, [1, nrof_features * dim_features])
    centers_2D = tf.tile(centers_1D, [nrof_features, 1])
    centers_3D = tf.reshape(centers_2D,[nrof_features, nrof_features, dim_features])
    features_3D = tf.reshape(features, [nrof_features, 1, dim_features])
    dist_inter_centers = features_3D - centers_3D
    dist_inter_centers_sum_dim = tf.reduce_sum(dist_inter_centers**2,2)/2
    centers_cts_batch_1D = tf.tile(centers_cts_batch,[nrof_features])
    centers_cts_batch_2D = tf.reshape(centers_cts_batch_1D, [nrof_features, nrof_features])
    dist_inter_centers_sum_unique = tf.div(dist_inter_centers_sum_dim, centers_cts_batch_2D)
    dist_inter_centers_sum_all = tf.reduce_sum(dist_inter_centers_sum_unique, 1)
    dist_inter_centers_sum = dist_inter_centers_sum_all - dist_centers_sum
    loss_inter_centers = tf.reduce_mean(dist_inter_centers_sum)

    ## total loss
    loss = tf.div(loss_center, loss_inter_centers)

    ## update centers
    diff = (1 - alfa) * (centers_batch - features)
#    ones = tf.ones_like(label, tf.float32)
#   centers_cts = tf.scatter_add(centers_cts, label, ones)  # counting the number of each class
#    centers_cts_batch = tf.gather(centers_cts, label)
    centers_cts_batch_reshape = tf.reshape(centers_cts_batch, [-1, 1])
    diff_mean = tf.div(diff, centers_cts_batch_reshape)
    centers = tf.scatter_sub(centers, label, diff_mean)
    zeros = tf.zeros_like(label, tf.float32)
    center_cts_clear = tf.scatter_update(centers_cts, label, zeros)
    # return loss, centers, label, centers_batch, diff, centers_cts, centers_cts_batch, diff_mean,center_cts_clear, centers_cts_batch_reshape
    return loss, centers,  loss_center, loss_inter_centers, center_cts_clear
    #return loss, centers, loss_center, loss_inter_centers, dist_inter_centers_sum_dim, centers_cts_batch_2D, dist_inter_centers_sum_unique, dist_inter_centers_sum_all, dist_inter_centers_sum, dist_inter_centers_sum, center_cts_clear




def class_level_triplet_loss_tf(features, nrof_samples, label, alfa, nrof_classes, beta, gamma): # tensorflow version
    """ Class_level_Triple_loss, triple loss implemented on the centers of the class intead of the individual sample
        --mzh 30072017s
    """
    dim_features = features.get_shape()[1].value
    centers = tf.get_variable('centers', [nrof_classes, dim_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    nrof_elements_per_class = tf.get_variable('centers_cts', [nrof_classes], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0), trainable=False)

    ## normalisation as the embedding vectors in order to similarity distance
    #features = tf.nn.l2_normalize(features, 1, 1e-10, name='feat_emb')

    ## calculate centers
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    diff_within = centers_batch - features
    dist_within = tf.reduce_sum(diff_within**2/2, axis=1, keep_dims=True)
    dist_within_center = tf.reduce_sum(dist_within, axis=0) ## sum all the elements in the dist_centers_sum, dist_within_center is a scale

    ## inter_center_loss calculation
    label_unique,idx = tf.unique(label)
    centers_batch_unique = tf.gather(centers,label_unique)#select the centers corresponding to the batch samples, otherwise the whole centers will cause the overflow of the centers_2D
    nrof_centers_batch_unique = tf.shape(centers_batch_unique)[0]##very important, tf.shape() can be used to get the run-time dynamic tensor shape; however .get_shape() can only be used to get the shape of the static shape of the tensor
    centers_1D = tf.reshape(centers_batch_unique, [1, nrof_centers_batch_unique * dim_features])
    centers_2D = tf.tile(centers_1D, [nrof_samples, 1])
    centers_3D = tf.reshape(centers_2D, [nrof_samples,nrof_centers_batch_unique, dim_features])
    features_3D = tf.reshape(features, [nrof_samples, 1, dim_features])
    dist_inter_centers = features_3D - centers_3D
    dist_inter_centers_sum_dim = tf.reduce_sum(dist_inter_centers**2,2)/2 # calculate the L2 of the features, [nrof_samples, nrof_classes, feature_dimension]
    dist_inter_centers_sum_all =  tf.reduce_sum(dist_inter_centers_sum_dim)#sum all the elements in the dist_inter_centers_sum_dim

    ## total loss
    dist_within_2D = tf.tile(dist_within, [1, nrof_centers_batch_unique])
    dist_matrix = dist_within_2D + beta*tf.ones([nrof_samples, nrof_centers_batch_unique]) - gamma*dist_inter_centers_sum_dim
    loss_matrix = tf.maximum(dist_matrix, tf.zeros([nrof_samples, nrof_centers_batch_unique], tf.float32))
    loss_pre = tf.reduce_sum(loss_matrix) - nrof_samples*beta
    #loss = tf.divide(loss_pre, nrof_samples)
    loss = tf.divide(loss_pre, tf.multiply(tf.cast(nrof_samples, tf.float32),
                                           tf.cast(nrof_centers_batch_unique, tf.float32) - tf.cast(1, tf.float32)))

    #centers = tf.scatter_sub(centers, label, diff)

    ##update centers
    zeros = tf.zeros_like(label_unique, tf.float32)
    ## calculation the repeat time of same label
    nrof_elements_per_class_clean = tf.scatter_update(nrof_elements_per_class, label_unique, zeros)
    ones = tf.ones_like(label, tf.float32)
    ## counting the number elments in each class, the class is in the order of the [0,1,2,3,....] as initialzation
    nrof_elements_per_class_update = tf.scatter_add(nrof_elements_per_class_clean, label, ones)
    ## nrof_elements_per_class_list is the number of the elements in each class in the batch
    nrof_elements_per_class_batch = tf.gather(nrof_elements_per_class_update, label)
    centers_cts_batch_reshape = tf.reshape(nrof_elements_per_class_batch, [-1, 1])
    diff_mean = tf.div(diff, centers_cts_batch_reshape)
    centers = tf.scatter_sub(centers, label, diff_mean)

    return loss, centers, dist_within_center, dist_inter_centers_sum_all, nrof_centers_batch_unique
















  


  







  










  






















