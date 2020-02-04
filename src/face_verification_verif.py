#### Function for face prediction ############
### Author: mingzuheng, 2018 #########
### copyright@zuheng ming, mingzuheng@gmail.com

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import sys


sys.path.append('../')
import align.face_align_mtcnn
import facenet_ext
import tensorflow as tf
import time



def load_models_forward_v2(args, Expr_dataset):
    if args.device == 'CPU':
        with tf.device('/cpu:0'):

            # Load the model of face detection
            pnet, rnet, onet = align.face_align_mtcnn.load_align_mtcnn(args.align_model_dir)


            ########### load face verif_expression model #############################
            # Load the model of face verification
            print('Model directory: %s' % args.model_dir)
            meta_file, ckpt_file = facenet_ext.get_model_filenames(os.path.expanduser(args.model_dir))

            sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            model_dir_exp = os.path.expanduser(args.model_dir)
            saver = tf.train.import_meta_graph(os.path.join(args.model_dir, meta_file))
            saver.restore(sess, os.path.join(model_dir_exp, ckpt_file))

            args_model.images_placeholder = tf.get_default_graph().get_tensor_by_name("image_batch:0")
            #args_model.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            args_model.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            #args_model.keep_probability_placeholder = tf.get_default_graph().get_tensor_by_name('keep_probability:0')
            args_model.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')


    elif args.device == 'GPU':
        with tf.device('/device:GPU:0'):
            # Load the model of face detection
            pnet, rnet, onet = align.face_align_mtcnn.load_align_mtcnn(args.align_model_dir)

            ########### load face verif_expression model #############################
            # Load the model of face verification
            print('Model directory: %s' % args.model_dir)
            meta_file, ckpt_file = facenet_ext.get_model_filenames(os.path.expanduser(args.model_dir))

            sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            model_dir_exp = os.path.expanduser(args.model_dir)
            saver = tf.train.import_meta_graph(os.path.join(args.model_dir, meta_file))
            saver.restore(sess, os.path.join(model_dir_exp, ckpt_file))

            args_model.images_placeholder = tf.get_default_graph().get_tensor_by_name("image_batch:0")
            args_model.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            #args_model.keep_probability_placeholder = tf.get_default_graph().get_tensor_by_name('keep_probability:0')
            args_model.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
    else:
        raise ValueError('Unknow device for computing! Only CPU or GPU can be chose!')

    return pnet, rnet, onet, sess, args_model



def face_embeddings(img_refs_, args, sess, args_model):

    # Load images
    image_size = args.image_size

    images = facenet_ext.load_data_im(img_refs_, False, False, image_size)
    if len(images.shape)==3:
        images = np.expand_dims(images, axis=0)

    feed_dict = {args_model.phase_train_placeholder: False, args_model.images_placeholder: images}
    #feed_dict = {args_model.phase_train_placeholder: False, args_model.images_placeholder: images, args_model.keep_probability_placeholder: 1.0}

    t2 = time.time()
    emb_array = sess.run([args_model.embeddings], feed_dict=feed_dict)
    t3 = time.time()
    print('Embedding calculation FPS:%d' % (int(1 / (t3 - t2))))

    return emb_array



def face_expression_multiref_forward(face_img_, emb_ref, args, sess, args_model):

    nrof_imgs = 1
    imgs = np.zeros((nrof_imgs, args.image_size, args.image_size, 3))
    imgs[0, :, :, :]=face_img_

    # Load images
    image_size = args.image_size

    images = facenet_ext.load_data_im(imgs, False, False, image_size)
    if len(images.shape) == 3:
        images = np.expand_dims(images,axis=0)

    feed_dict = {args_model.phase_train_placeholder: False, args_model.images_placeholder: images}
    #feed_dict = {args_model.phase_train_placeholder: False, args_model.images_placeholder: images, args_model.keep_probability_placeholder: 1.0}

    t2 = time.time()
    emb_array = sess.run([args_model.embeddings], feed_dict=feed_dict)
    t3 = time.time()
    print('Embedding calculation FPS:%d' % (int(1 / (t3 - t2))))
    embeddings1 = emb_array[0]
    embeddings2 = emb_ref[0]


    # Caculate the distance of embeddings and verification the two face
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    diff = np.subtract(embeddings1, embeddings2)
    if len(diff.shape)==2:
        dist = np.sum(np.square(diff), 1)
    elif len(diff.shape)==1:
        dist = np.sum(np.square(diff), 0)
    else:
        raise ValueError("Dimension of the embeddings2 is not correct!")


    predict_issame = np.less(dist, args.threshold)

    return predict_issame, dist



class args_model():
    def __init__(self):
        self.images_placeholder = None
        self.embeddings = None
        self.keep_probability_placeholder = None
        self.phase_train_placeholder = None
        self.logits = None
        self.phase_train_placeholder_expression = None