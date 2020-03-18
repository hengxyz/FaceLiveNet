### Face verification  #########


"""Functions for building the face recognition network.
"""
# MIT License
#### copyright at Auther: mingzuheng, 25/03/2019 #########

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import h5py
from sklearn.decomposition import PCA
import glob
import shutil
#import matplotlib.pyplot as plt
import csv
import cv2
import math
import glob
from numpy import linalg as LA
import imp
from tensorflow.python import pywrap_tensorflow
from tensorflow.python import debug as tf_debug

###### user custom lib
import facenet_ext
import lfw_ext
import metrics_loss
import train_BP
import evaluate_verif

class Args1:
    def __init__(self):
        self.model_dir = []
        self.pairs = []
        self.image_size = 0

args1 = Args1()

def main(args):
    
    module_networks = str.split(args.model_def,'/')[-1]
    network = imp.load_source(module_networks, args.model_def)  

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet_ext.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    random.seed(args.seed)


    #train_set = facenet_ext.get_dataset(args.data_dir)
    # train_set = facenet.get_huge_dataset(args.data_dir, args.trainset_start, args.trainset_end)
    # nrof_classes = len(train_set)
    
    # Get a list of image paths and their labels
    #image_list, label_list = facenet_ext.get_image_paths_and_labels(train_set)
    # image_list, label_list = facenet_ext.get_image_paths_and_labels_expression(train_set, args.labels_expression)
    image_list, label_list, classes = facenet_ext.get_image_paths_and_labels_gh(args.data_dir, 'train')
    #image_list_test, label_list_test, classes_test = facenet_ext.get_image_paths_and_labels_gh(args.data_dir, 'valid')
    image_list_ref, label_list_ref, classes_ref = facenet_ext.get_image_paths_and_labels_gh(args.data_dir, 'references')


    ### change labels from string to number
    classes_total = classes+classes_ref
    classes_total = list(set(classes_total))
    classes_total.sort()
    nrof_classes = len(classes_total)
    label_list = [classes_total.index(label) for label in label_list]
    label_list_ref = [classes_total.index(label) for label in label_list_ref]

    ### test
    for image in image_list:
        print('%s'%image)
        img = cv2.imread(image)
        if img is None:
            print (">>>>>>>>>>>>>>>>>>>>>>>> Image Error !")


    print('Total number of subjects: %d' % nrof_classes)
    print('Total number of images: %d' % len(image_list))

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        meta_file, ckpt_file = facenet_ext.get_model_filenames(os.path.expanduser(args.pretrained_model))
        print('Pre-trained model: %s' % pretrained_model)
        ### check the saved variables in ckpt file
        reader = pywrap_tensorflow.NewCheckpointReader(os.path.join(os.path.expanduser(args.pretrained_model), ckpt_file))
        var_to_shape_map = reader.get_variable_to_shape_map()

    
    if args.lfw_dir:
        print('LFW directory: %s' % args.lfw_dir)
        # Read the file containing the pairs used for testing
        pairs = lfw_ext.read_pairs(os.path.expanduser(args.lfw_pairs))
        # Get the paths for the corresponding images
        lfw_paths, lfw_actual_issame = lfw_ext.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)

    if args.valid_pairs:
        print('valid train samples directory: %s' % args.valid_pairs)
        valid_image_paths, actual_issame = read_valid_pairs(args.valid_pairs)

    if args.valid_pairs1:
        print('valid test samples directory: %s' % args.valid_pairs1)
        valid_image_paths1, actual_issame1 = read_valid_pairs(args.valid_pairs1)


    tf.set_random_seed(args.seed)
    global_step = tf.Variable(0, trainable=False)

    # Create a queue that produces indices into the image_list and label_list
    labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
    range_size = array_ops.shape(labels)[0]
    index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                         shuffle=True, seed=None, capacity=32)

    index_dequeue_op = index_queue.dequeue_many(args.batch_size*args.epoch_size, 'index_dequeue')

    learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
    # learning_rate_dyn_placeholder = tf.placeholder(tf.float32, name='learning_rate_dyn')

    batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

    # phase_train_placeholder_expression = tf.placeholder(tf.bool, name='phase_train_expression')


    image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')

    labels_id_placeholder = tf.placeholder(tf.int64, shape=(None,1), name='labels_id')

    # labels_expr_placeholder = tf.placeholder(tf.int64, shape=(None, 1), name='labels_expr')

    keep_probability_placeholder = tf.placeholder(tf.float32, name='keep_probability')

    input_queue = data_flow_ops.FIFOQueue(max(100000, args.batch_size*args.epoch_size),
                                dtypes=[tf.string, tf.int64],
                                shapes=[(1,), (1,)],
                                shared_name=None, name=None)
    enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_id_placeholder],
                                          name='enqueue_op')

    nrof_preprocess_threads = 4
    images_and_labels = []
    for _ in range(nrof_preprocess_threads):
        filenames, label_id = input_queue.dequeue()
       #filenames, label_id, label_expr = input_queue.dequeue()
       # filenames, label_id, label_expr = input_queue.dequeue_up_to()
        images = []
        #for filename in tf.unpack(filenames): ## tf0.12
        for filename in tf.unstack(filenames): ## tf1.0
            file_contents = tf.read_file(filename)
            #image = tf.image.decode_png(file_contents)
            image = tf.image.decode_jpeg(file_contents)
            if args.random_rotate:
                image = tf.py_func(facenet_ext.random_rotate_image, [image], tf.uint8)
            if args.random_crop:
                image = tf.random_crop(image, [args.image_size, args.image_size, 3])
            else:
                #image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                image = tf.image.resize_images(image, [args.image_size, args.image_size]) ## if input is face image, keep the whole image
            if args.random_flip:
                image = tf.image.random_flip_left_right(image)

            #pylint: disable=no-member
            image.set_shape((args.image_size, args.image_size, 3))
            ### whiten image
            images.append(tf.image.per_image_standardization(image))
        #images_and_labels.append([images, label_id, label_expr])
        images_and_labels.append([images, label_id])

    image_batch, label_batch_id = tf.train.batch_join(
        images_and_labels,
        batch_size=batch_size_placeholder,
        capacity=4 * nrof_preprocess_threads * args.batch_size,
        enqueue_many=True,
        shapes=[(args.image_size, args.image_size, 3), ()],
        allow_smaller_final_batch=True)
    #image_batch = tf.identity(image_batch, 'image_batch')
    image_batch = tf.identity(image_batch, 'input')
    label_id_batch = tf.identity(label_batch_id, 'label_id_batch')
    # label_expr_batch = tf.identity(label_batch_expr, 'label_expr_batch')


    print('Building training graph')

    # Build the inference graph
    prelogits, end_points = network.inference(image_batch, keep_probability_placeholder,
        phase_train=phase_train_placeholder, weight_decay=args.weight_decay)
    logits_id = slim.fully_connected(prelogits, nrof_classes, activation_fn=None, weights_initializer= tf.truncated_normal_initializer(stddev=0.1), weights_regularizer=slim.l2_regularizer(args.weight_decay),scope='Logits_verif', reuse=False)

    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

    # Add center loss
    if args.center_loss_factor>0.0:
        prelogits_center_loss_verif, prelogits_center_loss_verif_n, centers, _, centers_cts_batch_reshape, diff_mean \
            = metrics_loss.center_loss(embeddings, label_id_batch, args.center_loss_alfa, nrof_classes)


    cross_entropy_verif = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_id, labels=label_id_batch, name='cross_entropy_batch_verif')
    cross_entropy_mean_verif = tf.reduce_mean(cross_entropy_verif, name='cross_entropy_verif')

    #loss_verif_n = cross_entropy_verif + args.center_loss_factor*prelogits_center_loss_verif_n
    loss_verif_n = cross_entropy_verif + prelogits_center_loss_verif_n
    #loss_verif_n = prelogits_center_loss_verif_n
    #loss_verif_n = cross_entropy_verif
    loss_verif = tf.reduce_mean(loss_verif_n, name='loss_verif')

    loss_full = loss_verif


    # #### Training accuracy of softmax: check the underfitting or overfiting #############################
    correct_prediction_verif = tf.equal(tf.argmax(tf.exp(logits_id), 1), label_batch_id)
    softmax_acc_verif = tf.reduce_mean(tf.cast(correct_prediction_verif, tf.float32))
    ########################################################################################################

    learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                               args.learning_rate_decay_epochs * args.epoch_size,
                                               1.0, staircase=True)

    tf.summary.scalar('learning_rate', learning_rate)



    update_gradient_vars_expr = []
    update_gradient_vars_verif = []
    update_gradient_vars_mainstem = []
    update_gradient_vars_weights = []
    update_gradient_vars_networks = []

    for var in tf.trainable_variables():
        if 'Logits_lossweights/' in var.op.name:
            update_gradient_vars_weights.append(var)
        #Update variables for Branch Expression recogntion
        elif 'InceptionResnetV1_expression/' in var.op.name or 'Logits/' in var.op.name or 'Logits_0/' in var.op.name:
            print(var.op.name)
            update_gradient_vars_expr.append(var)
        # Update variables for Branch face verification
        elif 'InceptionResnetV1/Block8' in var.op.name or 'InceptionResnetV1/Repeat_2/block8' in var.op.name or 'Logits_verif/' in var.op.name:
            print(var.op.name)
            update_gradient_vars_verif.append(var)
        # Update variables for main stem
        else:
            print(var.op.name)
            update_gradient_vars_mainstem.append(var)

    paracnt, parasize = count_paras(update_gradient_vars_verif)
    print('The number of the updating parameters in the model Facenet is %dM, ......the size is : %dM bytes'%(paracnt/1e6, parasize/1e6))

    paracnt, parasize = count_paras(update_gradient_vars_expr)
    print('The number of the update parameters in the model Facial Expression is %dM, ......the size is : %dM bytes'%(paracnt/1e6, parasize/1e6))

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()


    #vars = update_gradient_vars_verif + update_gradient_vars_expr + update_gradient_vars_mainstem
    #vars = update_gradient_vars_verif+update_gradient_vars_mainstem
    vars = tf.trainable_variables()
    #vars = update_gradient_vars_verif
    train_op_mainstem, grads_full, grads_clip_full = train_BP.train(loss_full, global_step, args.optimizer,
        learning_rate, args.moving_average_decay, vars, summary_op, args.log_histograms)

    # Start running operations on the Graph.
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

    #sess = tf.Session(config=tf.ConfigProto(device_count={'CPU':1}, log_device_placement=True))
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    tf.train.start_queue_runners(sess=sess) ## wakeup the queue: start the queue operating defined by the tf.train.batch_join


    with sess.as_default():
        restore_vars_verif = []
        for var in tf.global_variables():
            # for var in restore_vars:
            # if 'InceptionResnetV1/' in var.op.name and var in tf.trainable_variables():
            #     restore_vars_verif.append(var)
            # if 'InceptionResnetV1/':
            #     restore_vars_verif.append(var)
            if var.op.name in var_to_shape_map:
                restore_vars_verif.append(var)
        restore_saver = tf.train.Saver(restore_vars_verif)
        restore_saver.restore(sess, os.path.join(os.path.expanduser(args.pretrained_model), ckpt_file))

        # Training and validation loop
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

        print('Running training')
        epoch = 0
        acc = 0
        val = 0
        far = 0
        acc1 = 0
        val1 = 0
        far1 = 0
        acc_lfw = 0
        val_lfw = 0
        far_lfw = 0

        best_acc_faceverif_lfw = 0
        best_acc_verif = 0
        best_acc_verif1 = 0

        with open(os.path.join(log_dir, 'LFW_result.txt'), 'at') as f:
            f.write('step, acc, val, far, best_acc\n')


        epoch_current = 0
        while epoch < args.max_nrof_epochs:
            epoch_current +=1
            step = sess.run(global_step, feed_dict=None)
            print('Epoch step: %d'%step)
            epoch = step // args.epoch_size

            ## Evaluate on valid-train dataset
            if (epoch % 1 == 0):
                if args.valid_pairs:
                    acc, val, far = evaluate(sess, phase_train_placeholder, batch_size_placeholder,
                                             embeddings, valid_image_paths, actual_issame, args.lfw_batch_size,
                                             args.lfw_nrof_folds,
                                             log_dir, step, summary_writer, args.evaluate_mode,
                                             keep_probability_placeholder, best_acc_verif, args, 'valid_pairs')
                ## saving the best_model for face verfication on valid-train samples
                if acc > best_acc_verif:
                    best_acc_verif = acc


                ## Evaluate on valid-test dataset
                if args.valid_pairs:
                    acc1, val1, far1 = evaluate(sess, phase_train_placeholder, batch_size_placeholder,
                                                embeddings, valid_image_paths1, actual_issame1, args.lfw_batch_size,
                                                args.lfw_nrof_folds,
                                                log_dir, step, summary_writer, args.evaluate_mode,
                                                keep_probability_placeholder, best_acc_verif, args, 'valid_pairs')
                ##
                if acc1 > best_acc_verif1:
                    best_acc_verif1 = acc1
                    best_model_dir = os.path.join(model_dir, 'best_model')
                    if not os.path.isdir(best_model_dir):  # Create the log directory if it doesn't exist
                        os.makedirs(best_model_dir)
                    if os.listdir(best_model_dir):
                        for file in glob.glob(os.path.join(best_model_dir, '*.*')):
                            os.remove(file)
                    for file in glob.glob(os.path.join(model_dir, '*.*')):
                        shutil.copy(file, best_model_dir)

            # ## Evaluate on LFW
            # if (epoch % 20 == 0):
            #     if args.lfw_dir:
            #         acc_lfw, val_lfw, far_lfw = evaluate(sess, phase_train_placeholder, batch_size_placeholder,
            #                               embeddings, lfw_paths, lfw_actual_issame, args.lfw_batch_size,
            #                               args.lfw_nrof_folds,
            #                               log_dir, step, summary_writer, args.evaluate_mode,
            #                               keep_probability_placeholder, best_acc_faceverif_lfw, args, 'LFW')
            #
            # ## saving the best_model for face verfication on LFW
            # if acc_lfw > best_acc_faceverif_lfw:
            #     best_acc_faceverif_lfw = acc_lfw

            # Train for one epoch
            step, softmax_acc_verif_, loss_verif_, cross_entropy_mean_verif_ \
                = train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder,
                        labels_id_placeholder, learning_rate_placeholder, phase_train_placeholder,
                        batch_size_placeholder, global_step, loss_verif, summary_op, summary_writer,
                        args.learning_rate_schedule_file, prelogits_center_loss_verif_n,
                        cross_entropy_mean_verif, acc, val, far, acc1, val1, far1, acc_lfw, val_lfw, far_lfw, keep_probability_placeholder, label_batch_id,  log_dir,
                        model_dir, softmax_acc_verif,  diff_mean,
                        centers, best_acc_verif, best_acc_verif1, best_acc_faceverif_lfw, train_op_mainstem, loss_full, epoch_current, grads_full, grads_clip_full, learning_rate, image_list_ref, label_list_ref, embeddings)




            # Save variables and the metagraph if it doesn't exist already
            save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

            ### test the evluate_verif.py
            # args1.model_dir = model_dir
            # args1.pairs = args.valid_pairs
            # args1.image_size = args.image_size
            # evaluate_verif.evaluate(args1)

    return model_dir
  
def train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder,
          labels_id_placeholder, learning_rate_placeholder, phase_train_placeholder,
          batch_size_placeholder, global_step, loss_verif, summary_op,  summary_writer,
          learning_rate_schedule_file, prelogits_center_loss_verif_n,
          cross_entropy_mean_verif, acc, val, far, acc1, val1, far1, acc_lfw, val_lfw, far_lfw, keep_probability_placeholder, label_batch_id, log_dir,
          model_dir, softmax_acc_verif, diff_mean,
          centers, best_acc_verif, best_acc_verif1, best_acc_faceverif_lfw, train_op_mainstem, loss_full, epoch_current, grads_full, grads_clip_full, learning_rate,
          image_list_ref, label_list_ref, embeddings):

    batch_number = 0
    
    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = facenet_ext.get_learning_rate_from_file(learning_rate_schedule_file, epoch_current)

    print('Index_dequeue_op....')
    index_epoch = sess.run(index_dequeue_op)
    label_id_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]

    ## adding the references images in the training data since reference - samples is heterogenous data has different modal ##
    percentage_reference = 0.5
    #percentage_reference = 1.0
    num_image_epoch = args.batch_size * args.epoch_size
    num_image_ref = int(num_image_epoch*percentage_reference)
    batch_image_ref = math.floor(num_image_ref/len(image_list_ref))
    label_id_epoch[:batch_image_ref*len(label_list_ref)]=np.repeat(np.array(label_list_ref),batch_image_ref)
    image_epoch[:batch_image_ref*len(image_list_ref)]=np.repeat(np.array(image_list_ref),batch_image_ref)
    index_0 =  list(range(num_image_epoch))
    random.shuffle(index_0)
    label_id_epoch = label_id_epoch[index_0]
    image_epoch = image_epoch[index_0]


    print('Enqueue__op....')
    # Enqueue one epoch of image paths and labels
    labels_id_array = np.expand_dims(np.array(label_id_epoch),1)
    image_paths_array = np.expand_dims(np.array(image_epoch),1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_id_placeholder: labels_id_array})

    # Training loop
    train_time = 0

    ####################### summing up the values the dimensions of the variables for checking the updating of the variables ##########################
    # Add validation loss and accuracy to summary
    summary = tf.Summary()

    with open(os.path.join(log_dir, 'loss.txt'), 'at') as f:
        f.write('loss_verif, cross_entropy_mean_verif\n')
    with open(os.path.join(log_dir, 'grads.txt'), 'at') as f:
        f.write('grads_weights,  grads_clip_weights\n')


    while batch_number < args.epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder: False, batch_size_placeholder: args.batch_size, keep_probability_placeholder: args.keep_probability}
        if (batch_number % 100 == 0):
            loss_verif_, step, prelogits_center_loss_verif_n_, cross_entropy_mean_verif_, \
            label_batch_id_, softmax_acc_verif_, diff_mean_, centers_, learning_rate_, _,  loss_full_, grads_full_, \
            grads_clip_full_,  embeddings_, summary_str \
                = sess.run([loss_verif, global_step, prelogits_center_loss_verif_n, cross_entropy_mean_verif,
                            label_batch_id, softmax_acc_verif, diff_mean, centers, learning_rate, train_op_mainstem,  loss_full, grads_full,
                            grads_clip_full, embeddings, summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step)
        else:
            loss_verif_, step, prelogits_center_loss_verif_n_, cross_entropy_mean_verif_, \
            label_batch_id_, softmax_acc_verif_, diff_mean_, centers_, learning_rate_, _, loss_full_, grads_full_, \
            grads_clip_full_ \
                = sess.run([loss_verif, global_step, prelogits_center_loss_verif_n, cross_entropy_mean_verif,
                            label_batch_id, softmax_acc_verif, diff_mean, centers, learning_rate, train_op_mainstem,  loss_full, grads_full,
                            grads_clip_full], feed_dict=feed_dict)
        print("step %d"%step)
        duration = time.time() - start_time




        ###############################################
        print('########## log dir is: %s '%log_dir )
        print('########## model dir is: %s ' %model_dir)
        print('Epoch: [%d][%d][%d/%d]\tTime %.3f\tLoss_verif %2.4f\tCrossEntropy_verif %2.4f\tsoftmax_acc_verif %2.4f\tCenterLoss_l2 %2.4f'  %
            (epoch, epoch_current, batch_number + 1, args.epoch_size, duration, loss_verif_, cross_entropy_mean_verif_, softmax_acc_verif_, np.mean(prelogits_center_loss_verif_n_)))
        print('Face verification on valid_pairs (samples-samples): acc %f, val %f, far %f, best_acc_verif %f'%(acc, val, far, best_acc_verif))
        print('Face verification on valid_pairs (reference-samples): acc1 %f, val1 %f, far1 %f, best_acc_verif1 %f'%(acc1, val1, far1, best_acc_verif1))
        print('Face verification on LFW: acc_LFW %f, val_LFW %f, far_LFW %f, best_acc_faceverif_LFW %f'%(acc_lfw, val_lfw, far_lfw, best_acc_faceverif_lfw))
        print('--------------------\n')
        print('centers:\n')
        for center in centers_:
            print('%f '%np.sum(center*center), end='')
        print('\n--------------------\n')
        print('diff_mean:\n')
        for diff in diff_mean_:
            print('%f '%np.sum(diff*diff), end='')
        print('\n--------------------\n')
        print('diff_embeddings:\n')
        loss_center_l2 = 0
        for i in range(len(embeddings_)):
            emb = embeddings_[i]
            label_i = label_batch_id_[i]
            center_i = centers_[label_i]
            diff = emb-center_i
            print('%f '%np.sum(diff*diff), end='')
            loss_center_l2 += np.sum(diff*diff)/2
        print('\nloss_center_l2: %f'%loss_center_l2)


        batch_number += 1
        train_time += duration

        #summary.value.add(tag='gradient/grad_total_norm', simple_value=grad_clip_sum)
        summary_writer.add_summary(summary, step)

        with open(os.path.join(log_dir, 'loss.txt'), 'at') as f:
            f.write('%f\t%f\n' % (loss_verif_, cross_entropy_mean_verif_))

            
            
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, step)

    return step, softmax_acc_verif_, loss_verif_, cross_entropy_mean_verif_

def evaluate(sess, phase_train_placeholder, batch_size_placeholder,
        embeddings, image_paths, actual_issame, batch_size, nrof_folds, log_dir, step, summary_writer,
             evaluate_mode, keep_probability_placeholder, best_acc, args, dataset):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Evaluating face verification %s...'%dataset)
    nrof_images = len(actual_issame) * 2
    nrof_batches = int(nrof_images / batch_size) ##floor division
    nrof_enque = batch_size*nrof_batches

    actual_issame = actual_issame[0:int(nrof_enque/2)]##left the elements in the final batch if it is not enough

    
    embedding_size = embeddings.get_shape()[1]

    emb_array = np.zeros((nrof_enque, embedding_size))
    lab_array = np.zeros((nrof_enque,))

    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    for ii in range(nrof_batches):
        print('batch %d, batch size %d' % (ii, batch_size))
        start_index = ii* batch_size
        end_index = min((ii + 1) * batch_size, nrof_images)
        paths_batch = image_paths[start_index:end_index]
        images = facenet_ext.load_data(paths_batch, False, False, args.image_size)

        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:batch_size, keep_probability_placeholder: 1.0, images_placeholder: images}
        emb = sess.run(embeddings, feed_dict=feed_dict)
        emb_array[start_index:end_index, :] = emb
        
    # assert np.array_equal(lab_array, np.arange(nrof_enque))==True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    if evaluate_mode == 'Euclidian':
        _, _, accuracy, val, val_std, far, fp_idx, fn_idx,best_threshold, val_threshold = lfw_ext.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)
    if evaluate_mode == 'similarity':
        pca = PCA(n_components=128)
        pca.fit(emb_array)
        emb_array_pca = pca.transform(emb_array)
        _, _, accuracy, val, val_std, far, fp_idx, fn_idx,best_threshold, val_threshold = lfw_ext.evaluate_cosine(emb_array_pca, actual_issame, nrof_folds=nrof_folds)
    for i in range(len(accuracy)):
        print('Accuracy: %1.3f@ best_threshold %1.3f' % (accuracy[i], best_threshold[i]))
    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='LFW/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='LFW/val_rate', simple_value=val)
    summary.value.add(tag='LFW/far_rate', simple_value=far)
    summary.value.add(tag='time/LFW', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir,'LFW_result.txt'),'at') as f:
        f.write('%d\t%.5f\t%.5f\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val, far, best_acc))

    acc = np.mean(accuracy)
    return acc, val, far


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)

def count_paras(vars):
    parasize = 0
    paracnt = 0
    for var in vars:
        print(var)
        paranum = 1
        for dim in var.get_shape():
            paranum *= dim.value
        parasize += paranum * sys.getsizeof(var.dtype)
        paracnt += paranum

    return paracnt, parasize

def read_valid_pairs(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        valid_image_paths = []
        actual_issame = []
        # random.shuffle(index)
        for line in lines:
            [label, img1, img2] = line.split(' ')
            actual_issame.append(bool(int(label)))
            valid_image_paths.append(img1)
            valid_image_paths.append(img2[:-1])

    return valid_image_paths, actual_issame


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='~/logs/')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='~/models/')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.',
        default='/data/zming/GH/face_recognition/models/20170131-234652') # #/data/zming/models/GH/face_recog/20191227-234749
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.')
    parser.add_argument('--data_dir_test', type=str,
        help='Path to the data directory containing aligned face for test. Multiple directories are separated with colon.')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.nn4')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=96)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_rotate', 
        help='Performs random rotations of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--decov_loss_factor', type=float,
        help='DeCov loss factor.', default=0.0)
    parser.add_argument('--center_loss_factor', type=float,
        help='Center loss factor.', default=0.0)
    parser.add_argument('--center_loss_alfa', type=float,
        help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--optimizer', type=str, choices=['Adagrad', 'Adadelta', 'Adam', 'RMSProp', 'Momentum', 'SGD'],
        help='The optimization algorithm to use', default='RMSProp')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_dyn', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=10)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=0.1)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
        help='Number of preprocessing (data loading and augumentation) threads.', default=4)
    parser.add_argument('--log_histograms', 
        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='../data/learning_rate_schedule.txt')
    parser.add_argument('--filter_filename', type=str,
        help='File containing image data used for dataset filtering', default='')
    parser.add_argument('--filter_percentile', type=float,
        help='Keep only the percentile images closed to its class center', default=100.0)
    parser.add_argument('--filter_min_nrof_images_per_class', type=int,
        help='Keep only the classes with this number of examples or more', default=0)
    parser.add_argument('--loss_weight_base1', type=float,
                        help='The base of the weight of the sub-loss in the full loss.', default=0)
    parser.add_argument('--loss_weight_base2', type=float,
                        help='The base of the weight of the second sub-loss in the full loss.', default=0)
    parser.add_argument('--valid_pairs', type=str,
                        help='The file containing the pairs to use for validation.',                             default='/data/zming/GH/face_recognition/data/face_samples-20191225T141159Z-001/face_samples/valid_train_pairs.txt') #valid_pairs.txt
    parser.add_argument('--valid_pairs1', type=str,
                        help='The file containing the pairs to use for validation.',                             default='/data/zming/GH/face_recognition/data/face_samples-20191225T141159Z-001/face_samples/valid_pairs.txt') #valid_pairs.txt

    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
                        help='The file containing the pairs to use for validation.', default='../data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
                        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_dir', type=str,
                        help='Path to the data directory containing aligned face patches.')
    parser.add_argument('--lfw_batch_size', type=int,
                        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--trainset_start', type=int,
        help='Number of the start of the train set', default=0)
    parser.add_argument('--trainset_end', type=int,
        help='Number of the end of the train set')
    parser.add_argument('--evaluate_mode', type=str,
                        help='The evaluation mode: Euclidian distance or similarity by cosine distance.',
                        default='Euclidian')


    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
