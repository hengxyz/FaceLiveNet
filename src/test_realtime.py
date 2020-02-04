#############   face_expression ######################################################################
#####  FER2013 :  0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral             ######
#####  CK+: 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise#  ######
#############   face_verification, face_expression ####################################################
import argparse
import os
import sys
import numpy as np
import cv2
from scipy import misc

import time


import align.face_align_mtcnn
import face_verification
import facenet_ext


#video = '../data/videoclips/WIN_20190627_15_31_21_Pro.mp4'
#video = '../data/videoclips/WIN_20190627_15_32_18_Pro.mp4'
#video = '../data/videoclips/WIN_20190627_16_06_54_Pro.mp4'

def verification_test(args):

    rect_len = 120
    offset_x = 50


    #Expr_str = ['Neutre', 'Colere', 'Degoute', 'Peur', 'Content', 'Triste', 'Surprise']  #####FER2013+ EXPRSSIONS_TYPE_fusion
    Expr_str = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    Expr_dataset = 'FER2013'

    c_red = (0, 0, 255)
    c_green = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    scale_size = 3  ## scale the original image as the input image to align the face


    ## load models for the face detection and verfication
    pnet, rnet, onet, sess, args_model = face_verification.load_models_forward_v2(args, Expr_dataset)


    face_img_refs_ = []
    img_ref_paths = []
    for img_ref_path in os.listdir(args.img_ref):
        img_ref_paths.append(img_ref_path)
        img_ref = misc.imread(os.path.join(args.img_ref, img_ref_path))  # python format
        img_size = img_ref.shape[0:2]

        bb, probs = align.face_align_mtcnn.align_mtcnn_realplay(img_ref, pnet, rnet, onet)
        if (bb == []):
            continue;

        bb_face = []
        probs_face = []
        for i, prob in enumerate(probs):
            if prob > args.face_detect_threshold:
                bb_face.append(bb[i])
                probs_face.append(prob)

        bb = np.asarray(bb_face)

        det = bb

        if det.shape[0] > 1:
            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = np.array(img_size) / 2
            offsets = np.vstack(
                [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            index = np.argmax(
                bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
            det = det[index, :]

        det = np.squeeze(det)
        x0 = det[0]
        y0 = det[1]


        bb_tmp = np.zeros(4, dtype=np.int32)
        bb_tmp[0] = np.maximum(det[0] - args.margin / 2, 0)
        bb_tmp[1] = np.maximum(det[1] - args.margin / 2, 0)
        bb_tmp[2] = np.minimum(det[2] + args.margin / 2, img_size[1])
        bb_tmp[3] = np.minimum(det[3] + args.margin / 2, img_size[0])

        face_img_ref = img_ref[bb_tmp[1]:bb_tmp[3], bb_tmp[0]:bb_tmp[2], :]
        face_img_ref = misc.imresize(face_img_ref, (args.image_size, args.image_size), interp='bilinear')
        face_img_ref_ = facenet_ext.load_data_im(face_img_ref, False, False, args.image_size)
        face_img_refs_.append(face_img_ref_)

        img_ref_cv = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
        cv2.rectangle(img_ref_cv, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), c_red, 2, 8, 0)
        img_ref_name = img_ref_path.split('.')[0]
        cv2.putText(img_ref_cv, "%s" % img_ref_name, (int(x0), int(y0 - 10)), font,
                    1,
                    c_red, 2)
        cv2.imshow('%s'%img_ref_path, img_ref_cv)
        cv2.waitKey(20)

    face_img_refs_ = np.array(face_img_refs_)

    emb_ref = face_verification.face_embeddings(face_img_refs_, args, sess, args_model, Expr_dataset)


    ################ capture the camera for realplay #############################################
    if args.video == '0':
        video = 0
    else:
        video = args.video
    cap = cv2.VideoCapture(video)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # fourcc = cv2.CV_FOURCC('M', 'J', 'P', 'G')
    # out = cv2.VideoWriter('output.avi', fourcc, 20, (600,800))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 5, (frame_height, frame_width))

    realplay_window = "Realplay"
    cv2.namedWindow(realplay_window, cv2.WINDOW_NORMAL)


    while (True):
        if cv2.getWindowProperty(realplay_window, cv2.WINDOW_NORMAL) < 0:
            return
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret==False:
            break;



        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        im_np = cv2_im
        img_size = im_np.shape[0:2]
        im_np_scale = cv2.resize(im_np, (int(img_size[1] / scale_size), int(img_size[0] / scale_size)),
                                  interpolation=cv2.INTER_LINEAR)
        bb, probs = align.face_align_mtcnn.align_mtcnn_realplay(im_np_scale, pnet, rnet, onet)

        bb_face = []
        probs_face = []
        for i, prob in enumerate(probs):
            if prob > args.face_detect_threshold:
                bb_face.append(bb[i])
                probs_face.append(prob)

        bb = np.asarray(bb_face)
        probs = np.asarray(probs_face)

        bb = bb*scale_size #re_scale of the scaled image for align_face

        if (len(bb) > 0):
            for i in range(bb.shape[0]):
                prob = probs[i]
                det = bb[i]
                bb_tmp = np.zeros(4, dtype=np.int32)
                bb_tmp[0] = np.maximum(det[0] - args.margin / 2, 0)
                bb_tmp[1] = np.maximum(det[1] - args.margin / 2, 0)
                bb_tmp[2] = np.minimum(det[2] + args.margin / 2, img_size[1])
                bb_tmp[3] = np.minimum(det[3] + args.margin / 2, img_size[0])

                face_img = im_np[bb_tmp[1]:bb_tmp[3], bb_tmp[0]:bb_tmp[2], :]
                face_img_ = misc.imresize(face_img, (args.image_size, args.image_size), interp='bilinear')
                face_img_ = facenet_ext.load_data_im(face_img_, False, False, args.image_size)


                #########
                x0 = bb[i][0]
                y0 = bb[i][1]
                x1 = bb[i][2]
                y1 = bb[i][3]
                offset_y = int((y1-y0)/7)

                # face experssion
                ##### 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise   ############
                t2 = time.time()
                predict_issames, dists, express_probs = face_verification.face_expression_multiref_forward(face_img_, emb_ref, args, sess, args_model, Expr_dataset)
                t3 = time.time()


                print('face verif FPS:%d' % (int(1 / ((t3 - t2)))))

                predict_issame_idxes = [i for i, predict_issame in enumerate(predict_issames) if predict_issame == True]

                if len(predict_issame_idxes)>1:
                    predict_issame_idx = np.argmin(dists)
                elif len(predict_issame_idxes) == 1:
                    predict_issame_idx = predict_issame_idxes[0]


                if len(predict_issame_idxes):
                    i = predict_issame_idx
                    dist = dists[i]
                    img_ref_name = img_ref_paths[i].split('.')[0]

                    cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), c_green, 2,
                                  8,
                                  0)
                    cv2.putText(frame, "%.4f" % prob, (int(x0), int(y0)), font,
                                0.5,
                                c_green, 1)
                    cv2.putText(frame, "%.2f" % dist, (int(x1), int(y1)), font,
                                0.5,
                                c_green, 1)
                    cv2.putText(frame, "%s" % img_ref_name, (int((x1 + x0) / 2), int(y0 - 10)), font,
                                1,
                                c_green, 2)

                    for k in range(express_probs.shape[0]):
                        cv2.putText(frame, Expr_str[k],
                                    (int(x1 + offset_x / 4), int(y0 + offset_y * k + offset_y / 4)),
                                    font,
                                    0.5,
                                    c_green, 1)
                        cv2.rectangle(frame, (
                        int(x1 + offset_x / 4), int(y0 + offset_y * k + offset_y / 4 + offset_y / 5)),
                                      (int(x1 + offset_x / 4 + rect_len * express_probs[k]),
                                       int(y0 + offset_y * k + + offset_y / 4 + offset_y / 2)),
                                      c_green, cv2.FILLED,
                                      8,
                                      0)
                else:
                    dist = min(dists)
                    cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), c_red, 2,
                                  8,
                                  0)
                    cv2.putText(frame, "%.4f" % prob, (int(x0), int(y0)), font,
                                0.5,
                                c_red, 1)
                    cv2.putText(frame, "%.2f" % dist, (int(x1), int(y1)), font,
                                0.5,
                                c_red, 1)

                    for k in range(express_probs.shape[0]):
                        cv2.putText(frame, Expr_str[k], (int(x1 + offset_x / 4), int(y0 + offset_y * k + offset_y / 4)),
                                    font,
                                    0.5,
                                    c_red, 1)
                        cv2.rectangle(frame, (int(x1 + offset_x / 4), int(y0 + offset_y * k + offset_y / 4 + offset_y / 5)),
                                      (int(x1 + offset_x / 4 + rect_len * express_probs[k]),
                                       int(y0 + offset_y * k + + offset_y / 4 + offset_y / 2)),
                                      c_red, cv2.FILLED,
                                      8,
                                      0)




        # visualation
        out.write(frame)
        cv2.imshow(realplay_window, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    out.release()
    cap.release()
    cv2.destroyAllWindows()


    return

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str,
                        help='The device used for computing, the default device is CPU', default='CPU')
    parser.add_argument('--img_ref', type=str, help='Directory with unaligned image 1.', default='../data/images')
    parser.add_argument('--video', type=str, help='The video to be detected. The default valude is 0 to capture the video from camera.', default='0')

    ## face_align_mtcnn_test() arguments
    parser.add_argument('--align_model_dir', type=str,
                        help='Directory containing the models for the face detection', default='./models')
    parser.add_argument('--model_dir', type=str,
                        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters',
                        default='../models/20180115-025629_model/best_model') #20170920-174400_expression') #../model/20170501-153641#20161217-135827#20170131-234652
    parser.add_argument('--threshold', type=float,
                        help='The threshold for the face verification',default=0.9)
    parser.add_argument('--face_detect_threshold', type=float,
                        help='The threshold for the face detection', default=0.9)

    ## face_align_mtcnn_test() arguments
    parser.add_argument('--output_dir', type=str, help='Directory with aligned face thumbnails.', default='./align/output')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=32)
    parser.add_argument('--random_order',
                        help='Shuffles the order of images to enable alignment using multiple processes.',
                        action='store_true')


    return parser.parse_args(argv)

if __name__ == '__main__':
    verification_test(parse_arguments(sys.argv[1:]))