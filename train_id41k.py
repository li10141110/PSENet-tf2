import sys
import tensorflow as tf
import argparse
import numpy as np
import shutil
import os

from dataset import CTW1500Loader, ctw_train_loader
from metrics import runningScore
import models
from util import Logger
import time
import util


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def ohem_single(score, gt_text, training_mask):
    pos_num = (int)(np.sum(gt_text > 0.5)) - (int)(np.sum((gt_text > 0.5) & (training_mask <= 0.5)))
    
    if pos_num == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask
    
    neg_num = (int)(np.sum(gt_text <= 0.5))
    neg_num = (int)(min(pos_num * 3, neg_num))
    
    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask
    #print()
    neg_score = score[gt_text <= 0.5]
    neg_score_sorted = np.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]

    selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
    selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
    return selected_mask
    
def ohem_batch(scores, gt_texts, training_masks):
    scores = scores.numpy()
    gt_texts = gt_texts.numpy()
    training_masks = training_masks.numpy()

    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

    selected_masks = np.concatenate(selected_masks, 0)
    #selected_masks = torch.from_numpy(selected_masks).float()
    selected_masks = tf.convert_to_tensor(selected_masks,dtype=tf.float32)

    return selected_masks
    
    
def dice_loss(input, target, mask):
    #input = torch.sigmoid(input)
    input = tf.sigmoid(input)

    #input = input.contiguous().view(input.size()[0], -1)
    #target = target.contiguous().view(target.size()[0], -1)
    #mask = mask.contiguous().view(mask.size()[0], -1)
    input = tf.reshape(input, (input.shape[0], -1))
    target = tf.reshape(target, (target.shape[0], -1))
    mask = tf.reshape(mask, (mask.shape[0], -1))
    
    
    input = input * mask
    target = target * mask

    #a = torch.sum(input * target, 1)
    #b = torch.sum(input * input, 1) + 0.001
    #c = torch.sum(target * target, 1) + 0.001
    a = tf.reduce_sum(input * target, 1)
    b = tf.reduce_sum(input * input, 1) + 0.001
    c = tf.reduce_sum(target * target, 1) + 0.001
    
    
    d = (2 * a) / (b + c)
    #dice_loss = torch.mean(d)
    dice_loss = tf.reduce_mean(d)
    return 1 - dice_loss
    
    
def cal_text_score(texts, gt_texts, training_masks, running_metric_text):
    training_masks = training_masks.numpy()
    pred_text = tf.sigmoid(texts).numpy() * training_masks
    pred_text[pred_text <= 0.5] = 0
    pred_text[pred_text >  0.5] = 1
    pred_text = pred_text.astype(np.int32)
    gt_text = gt_texts.numpy() * training_masks
    gt_text = gt_text.astype(np.int32)
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text

def cal_kernel_score(kernels, gt_kernels, gt_texts, training_masks, running_metric_kernel):
    mask = (gt_texts * training_masks).numpy()
    kernel = kernels[:, -1, :, :]
    gt_kernel = gt_kernels[:, -1, :, :]
    #pred_kernel = torch.sigmoid(kernel).data.cpu().numpy()
    pred_kernel = tf.sigmoid(kernel).numpy()
    pred_kernel[pred_kernel <= 0.5] = 0
    pred_kernel[pred_kernel >  0.5] = 1
    pred_kernel = (pred_kernel * mask).astype(np.int32)
    gt_kernel = gt_kernel.numpy()
    gt_kernel = (gt_kernel * mask).astype(np.int32)
    running_metric_kernel.update(gt_kernel, pred_kernel)
    score_kernel, _ = running_metric_kernel.get_scores()
    return score_kernel
    
    
def train(train_loader, model, criterion, optimizer, epoch):
    #model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    running_metric_text = runningScore(2)
    running_metric_kernel = runningScore(2)
    
    end = time.time()
    for batch_idx, (imgs, gt_texts, gt_kernels, training_masks, data_length) in enumerate(train_loader):
        
        with tf.GradientTape() as tape:
        
            data_time.update(time.time() - end)
        
            outputs = model(imgs)
            outputs = tf.transpose(outputs,(0,3,1,2))
            texts = outputs[:, 0, :, :]
            kernels = outputs[:, 1:, :, :]
        
            selected_masks = ohem_batch(texts, gt_texts, training_masks)
        
            loss_text = criterion(texts, gt_texts, selected_masks)
        
            loss_kernels = []
            mask0 = tf.sigmoid(texts).numpy()
            mask1 = training_masks.numpy()
            selected_masks = ((mask0 > 0.5) & (mask1 > 0.5)).astype('float32')
            #selected_masks = torch.from_numpy(selected_masks).float()
            selected_masks = tf.convert_to_tensor(selected_masks,dtype=tf.float32)
            #selected_masks = Variable(selected_masks.cuda())
        
            for i in range(6):
                kernel_i = kernels[:, i, :, :]
                gt_kernel_i = gt_kernels[:, i, :, :]
                loss_kernel_i = criterion(kernel_i, gt_kernel_i, selected_masks)
                loss_kernels.append(loss_kernel_i)
            loss_kernel = sum(loss_kernels) / len(loss_kernels)
        
            loss = 0.7 * loss_text + 0.3 * loss_kernel
        #反向计算各层loss
        losses.update(loss.numpy(), imgs.shape[0])

        #计算梯度 tape模式，保持跟踪
        grads = tape.gradient(loss, model.trainable_weights)
        #
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        score_text = cal_text_score(texts, gt_texts, training_masks, running_metric_text)
        score_kernel = cal_kernel_score(kernels, gt_kernels, gt_texts, training_masks, running_metric_kernel)
        
        batch_time.update(time.time() - end)
        end = time.time()
        size = data_length / args.batch_size
        if batch_idx % 20 == 0:
            output_log  = '({batch}/{size}) Batch: {bt:.3f}s | TOTAL: {total:.0f}min \
            | ETA: {eta:.0f}min | Loss: {loss:.4f} | Acc_t: {acc: .4f} | IOU_t: {iou_t: .4f}\
             | IOU_k: {iou_k: .4f}'.format(
                batch=batch_idx + 1,
                #size=len(train_loader),
                size=data_length/args.batch_size,
                bt=batch_time.avg,
                total=batch_time.avg * batch_idx / 60.0,
                #eta=batch_time.avg * (len(train_loader) - batch_idx) / 60.0,
                eta=batch_time.avg * (size - batch_idx) / 60.0,
                loss=losses.avg,
                acc=score_text['Mean Acc'],
                iou_t=score_text['Mean IoU'],
                iou_k=score_kernel['Mean IoU'])
            print(output_log)
            sys.stdout.flush()
            
    return (losses.avg, score_text['Mean Acc'], score_kernel['Mean Acc'], score_text['Mean IoU'], score_kernel['Mean IoU'])
    

def get_new_optimizer(args, optimizer, epoch):
    global state
    if epoch in args.schedule:
        args.lr = args.lr * 0.1
        conf = optimizer.get_config()
        conf['learning_rate'] = args.lr
        optimizer = optimizer.from_config(conf)
    return optimizer


def main(args):
    if args.checkpoint == '':
        args.checkpoint = "checkpoints/ctw1500_%s_bs_%d_ep_%d"%(args.arch, args.batch_size, args.n_epoch)
    if args.pretrain:
        if 'synth' in args.pretrain:
            args.checkpoint += "_pretrain_synth"
        else:
            args.checkpoint += "_pretrain_ic17"

    print('checkpoint path: %s'%args.checkpoint)
    print('init lr: %.8f'%args.lr)
    print('schedule: ', args.schedule)
    sys.stdout.flush()

    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    kernel_num = 7
    min_scale = 0.4
    start_epoch = 0
    
    data_loader = CTW1500Loader(is_transform=True, img_size=args.img_size, kernel_num=kernel_num, min_scale=min_scale)
    #train_loader = ctw_train_loader(data_loader, batch_size=args.batch_size)
    
    if args.arch == "resnet50":
        model = models.resnet50(pretrained=True, num_classes=kernel_num)
    elif args.arch == "resnet101":
        model = models.resnet101(pretrained=True, num_classes=kernel_num)
    elif args.arch == "resnet152":
        model = models.resnet152(pretrained=True, num_classes=kernel_num)

    #resnet18 and 34 didn't inplement pretrained
    elif args.arch == "resnet18":
        model = models.resnet18(pretrained=False, num_classes=kernel_num)
    elif args.arch == "resnet34":
        model = models.resnet34(pretrained=False, num_classes=kernel_num)


    elif args.arch == "mobilenetv2":
        model = models.resnet152(pretrained=True, num_classes=kernel_num)
    elif args.arch == "mobilenetv3large":
        model = models.mobilenetv3_large(pretrained=False, num_classes=kernel_num)

    elif args.arch == "mobilenetv3small":
        model = models.mobilenetv3_small(pretrained=False, num_classes=kernel_num)

    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0.99, decay=5e-4)
    
    title = 'CTW1500'
    if args.pretrain:
        print('Using pretrained model.')
        assert os.path.isfile(args.pretrain), 'Error: no checkpoint directory found!'
        
        
        
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss','Train Acc.', 'Train IOU.'])
    elif args.resume:
        print('Resuming from checkpoint.')
        
        model.load_weights(args.resume)
        
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        print('Training from scratch.')
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss','Train Acc.', 'Train IOU.'])
        
    for epoch in range(start_epoch, args.n_epoch):
        optimizer = get_new_optimizer(args, optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.n_epoch, optimizer.get_config()['learning_rate']))
        
        train_loader = ctw_train_loader(data_loader, batch_size=args.batch_size)
        
        train_loss, train_te_acc, train_ke_acc, train_te_iou, train_ke_iou = train(train_loader, model, dice_loss,\
                                                                                   optimizer, epoch)
        
        model.save_weights('%s%s' % (args.checkpoint, '/model_tf/weights'))
        
        logger.append([optimizer.get_config()['learning_rate'], train_loss, train_te_acc, train_te_iou])
    logger.close()
    

def set_gpu_memory_growth():
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
		try:
			# 设置 GPU 显存占用为按需分配
			for gpu in gpus:
				tf.config.experimental.set_memory_growth(gpu, True)
			logical_gpus = tf.config.experimental.list_logical_devices('GPU')
			print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
		except RuntimeError as e:
			# 异常处理
			print(e)
	else :
		print('No GPU')
        
        
def limit_gpu_memory():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2524)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    #parser.add_argument('--arch', nargs='?', type=str, default='mobilenetv3small')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet50')

#todo img_size as an effect para
    parser.add_argument('--img_size', nargs='?', type=int, default=640,
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=600, 
                        help='# of the epochs')
    parser.add_argument('--schedule', type=int, nargs='+', default=[200, 400],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch Size')
    parser.add_argument('--lr', nargs='?', type=float, default=1e-3, 
                        help='Learning Rate')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--pretrain', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
    args = parser.parse_args()
    
    #set_gpu_memory_growth()
    limit_gpu_memory()
    
    main(args)