import os
import cv2
import sys
import time
import collections
import argparse
import numpy as np
import tensorflow as tf
from dataset import CTW1500TestLoader, ctw_test_loader
import models
import util
# c++ version pse based on opencv 3+
#from pse import pse
# python pse
from pypse import pse as pypse

def extend_3c(img):
    img = img.reshape(img.shape[0], img.shape[1], 1)
    img = np.concatenate((img, img, img), axis=2)
    return img

def debug(idx, img_paths, imgs, output_root):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    
    col = []
    for i in range(len(imgs)):
        row = []
        for j in range(len(imgs[i])):
            # img = cv2.copyMakeBorder(imgs[i][j], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
            row.append(imgs[i][j])
        res = np.concatenate(row, axis=1)
        col.append(res)
    res = np.concatenate(col, axis=0)
    img_name = img_paths[idx].split('/')[-1]
    print(idx, '/', len(img_paths), img_name)
    cv2.imwrite(output_root + img_name, res)







def write_result_as_txt(image_name, bboxes, path):
    if not os.path.exists(path):
        os.makedirs(path)

    filename = util.io.join_path(path, '%s.txt'%(image_name))
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        # line = "%d, %d, %d, %d, %d, %d, %d, %d\n"%tuple(values)
        line = "%d"%values[0]
        for v_id in range(1, len(values)):
            line += ", %d"%values[v_id]
        line += '\n'
        lines.append(line)
    util.io.write_lines(filename, lines)

def polygon_from_points(points):
    """
    Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
    """
    resBoxes=np.empty([1, 8],dtype='int32')
    resBoxes[0, 0] = int(points[0])
    resBoxes[0, 4] = int(points[1])
    resBoxes[0, 1] = int(points[2])
    resBoxes[0, 5] = int(points[3])
    resBoxes[0, 2] = int(points[4])
    resBoxes[0, 6] = int(points[5])
    resBoxes[0, 3] = int(points[6])
    resBoxes[0, 7] = int(points[7])
    pointMat = resBoxes[0].reshape([2, 4]).T
    return plg.Polygon(pointMat)


    
    
def test(args):
    data_loader = CTW1500TestLoader(long_size=args.long_size)
    test_loader = ctw_test_loader(data_loader, 1)
    
    # Setup Model
    if args.arch == "resnet50":
        model = models.resnet50(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "resnet101":
        model = models.resnet101(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "resnet152":
        model = models.resnet152(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "resnet18":
        model = models.resnet18(pretrained=True, num_classes=7, scale=args.scale)
    if args.resume is not None:          
        print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
        model.load_weights(args.resume)
        print("Loaded checkpoint '{}' ".format(args.resume,))
        sys.stdout.flush()
    else:
        print("No checkpoint found at '{}'".format(args.resume))
        sys.stdout.flush()
            
    total_frame = 0.0
    total_time = 0.0
    for idx, (org_img, img, data_length) in enumerate(test_loader):
        print('progress: %d / %d'%(idx, data_length))
        sys.stdout.flush()
        
        org_img = org_img.numpy().astype('uint8')[0]
        text_box = org_img.copy()
        
        start = time.time()
        outputs = model(img)
        outputs = tf.transpose(outputs,(0,3,1,2))
        
        score = tf.sigmoid(outputs[:, 0, :, :])
        outputs = (tf.sign(outputs - args.binary_th) + 1) / 2
        
        text = outputs[:, 0, :, :]
        kernels = outputs[:, 0:args.kernel_num, :, :] * text
        
        score = score.numpy()[0].astype(np.float32)
        text = text.numpy()[0].astype(np.uint8)
        kernels = kernels.numpy()[0].astype(np.uint8)
        
        # c++ version pse  #编译问题 暂时不用
        #pred = pse(kernels, args.min_kernel_area / (args.scale * args.scale))
        # python version pse
        pred = pypse(kernels, args.min_kernel_area / (args.scale * args.scale))
        
        # scale = (org_img.shape[0] * 1.0 / pred.shape[0], org_img.shape[1] * 1.0 / pred.shape[1])
        scale = (org_img.shape[1] * 1.0 / pred.shape[1], org_img.shape[0] * 1.0 / pred.shape[0])
        label = pred
        label_num = np.max(label) + 1
        bboxes = []
        for i in range(1, label_num):
            points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

            if points.shape[0] < args.min_area / (args.scale * args.scale):
                continue

            score_i = np.mean(score[label == i])
            if score_i < args.min_score:
                continue

            # rect = cv2.minAreaRect(points)
            binary = np.zeros(label.shape, dtype='uint8')
            binary[label == i] = 1

            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour = contours[0]
            # epsilon = 0.01 * cv2.arcLength(contour, True)
            # bbox = cv2.approxPolyDP(contour, epsilon, True)
            bbox = contour

            if bbox.shape[0] <= 2:
                continue
            
            # bbox = bbox * scale
            # bbox = bbox.astype('int32')
            # bboxes.append(bbox.reshape(-1))
            bbox = bbox * scale
            bbox = cv2.minAreaRect(bbox.reshape((-1, 2)).astype(np.float32))  #
            bbox = cv2.boxPoints(bbox)  #
            bbox = bbox.astype('int32')
            bboxes.append(bbox.reshape(-1))

        end = time.time()
        total_frame += 1
        total_time += (end - start)
        print('fps: %.2f'%(total_frame / total_time))
        sys.stdout.flush()

        for bbox in bboxes:
            cv2.drawContours(text_box, [bbox.reshape(bbox.shape[0] // 2, 2)], -1, (0, 255, 0), 2)

        image_name = data_loader.img_paths[idx].split('/')[-1].split('.jpg')[0]
        write_result_as_txt(image_name, bboxes, 'outputs/submit_ctw1500/')
        
        text_box = cv2.resize(text_box, (text.shape[1], text.shape[0]))
        debug(idx, data_loader.img_paths, [[text_box]], 'outputs/vis_ctw1500/')
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet18')
    parser.add_argument('--resume', nargs='?', type=str, default='checkpoints/',    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--binary_th', nargs='?', type=float, default=1.0,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--kernel_num', nargs='?', type=int, default=3,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--scale', nargs='?', type=int, default=1,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--long_size', nargs='?', type=int, default=1280,
                        help='')
    parser.add_argument('--min_kernel_area', nargs='?', type=float, default=10.0,
                        help='min kernel area')
    parser.add_argument('--min_area', nargs='?', type=float, default=300.0,
                        help='min area')
    parser.add_argument('--min_score'test_ctw1500.py, nargs='?', type=float, default=0.93,
                        help='min score')
    
    args = parser.parse_args()
    test(args)