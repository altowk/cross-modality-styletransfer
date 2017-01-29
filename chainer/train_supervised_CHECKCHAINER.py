from __future__ import print_function
import numpy as np
import os, re
import argparse
from PIL import Image

from chainer import cuda, Variable, optimizers, serializers
from net import *

#Used for live loss plot
import pylab as plt

import math

# for calcutaing time elapse
import time

def load_image(path, size):
    image = Image.open(path).convert('RGB')
    w,h = image.size
    if w < h:
        if w < size:
            image = image.resize((size, int(math.ceil(size*h/w))))
            w, h = image.size
    else:
        if h < size:
            image = image.resize((int(math.ceil(size*w/h)), size))
            w, h = image.size
    image = image.crop(((w-size)*0.5, (h-size)*0.5, (w+size)*0.5, (h+size)*0.5))
    return xp.asarray(image, dtype=np.float32).transpose(2, 0, 1)

def gram_matrix(y):
    b, ch, h, w = y.data.shape
    features = F.reshape(y, (b, ch, w*h))
    gram = F.batch_matmul(features, features, transb=True)/np.float32(ch*w*h)
    return gram

def total_variation(x):
    xp = cuda.get_array_module(x.data)
    b, ch, h, w = x.data.shape
    wh = Variable(xp.asarray([[[[1], [-1]], [[0], [0]], [[0], [0]]], [[[0], [0]], [[1], [-1]], [[0], [0]]], [[[0], [0]], [[0], [0]], [[1], [-1]]]], dtype=np.float32), volatile=x.volatile)
    ww = Variable(xp.asarray([[[[1, -1]], [[0, 0]], [[0, 0]]], [[[0, 0]], [[1, -1]], [[0, 0]]], [[[0, 0]], [[0, 0]], [[1, -1]]]], dtype=np.float32), volatile=x.volatile)
    return F.sum(F.convolution_2d(x, W=wh) ** 2) + F.sum(F.convolution_2d(x, W=ww) ** 2)

parser = argparse.ArgumentParser(description='Real-time style transfer')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--dataset', '-d', default='dataset', type=str,
                    help='dataset directory path (according to the paper, use MSCOCO 80k images)')
parser.add_argument('--style_image', '-s', type=str, required=False,
                    help='style image path')
parser.add_argument('--batchsize', '-b', type=int, default=1,
                    help='batch size (default value is 1)')
parser.add_argument('--initmodel', '-i', default=None, type=str,
                    help='initialize the model from given file')
parser.add_argument('--resume', '-r', default=None, type=str,
                    help='resume the optimization from snapshot')
parser.add_argument('--output', '-o', default=None, type=str,
                    help='output model file path without extension')
parser.add_argument('--lambda_tv', default=1e-6, type=float,
                    help='weight of total variation regularization according to the paper to be set between 10e-4 and 10e-6.')
parser.add_argument('--lambda_feat', default=1.0, type=float)
parser.add_argument('--lambda_style', default=5.0, type=float)
parser.add_argument('--epoch', '-e', default=5, type=int)
parser.add_argument('--lr', '-l', default=1e-3, type=float)
parser.add_argument('--checkpoint', '-c', default=0, type=int)
parser.add_argument('--image_size', default=256, type=int)

## Added
parser.add_argument('--groundtruth', default='groundtruth', type=str,
                    help='groundtruth images must have same name as training image')
parser.add_argument('--folder', '-f', default='', type=str,
                    help='output folder for model, loss files')
parser.add_argument('--width', default=181, type=int)
parser.add_argument('--height', default=217, type=int)
parser.add_argument('--feature_layer', default=2, type=int)
parser.add_argument('--validate', default=0, type=int)
parser.add_argument('--validation_set', nargs=2, type=str,
                    help='--validation_set [input] [ground_truth]')
args = parser.parse_args()

print(args)

# Added
width = args.width
height = args.height
# output_folder = args.folder
feature_layer = args.feature_layer
validate = args.validate # 0 for trianing, 1 for validation


image_size = args.image_size
# n_epoch = args.epoch
lambda_tv = args.lambda_tv
lambda_f = args.lambda_feat
lambda_s = args.lambda_style
# style_prefix, _ = os.path.splitext(os.path.basename(args.style_image))
# output = style_prefix if args.output == None else args.output

# nameing prefix of model and state files
output = 'f{}_s{}_tv{}_lf{}'.format(lambda_f, lambda_s, lambda_tv, feature_layer)

# name of directory to put everything in
output_folder = args.folder if args.folder else 'model_f{}_s{}_tv{}_lf{}/'.format(lambda_f, lambda_s, lambda_tv, feature_layer)

model = FastStyleNet()

if (validate == 1):
    batchsize = 1
    n_epoch = 1
    # Create output folder
    if output_folder != '':
        # Check if output folder exist
        if not os.path.isdir(output_folder):
            print("Creating dir {}".format(os.path.abspath(output_folder)))
            os.mkdir(output_folder)
        else:
            print("folder {} exists".format((output_folder)))
else:
    batchsize = args.batchsize
    n_epoch = args.epoch
    # Create output folder
    if output_folder != '':
        # Check if output folder exist
        if not os.path.isdir(output_folder):
            print("Creating dir {}".format(os.path.abspath(output_folder)))
            os.mkdir(output_folder)
            os.mkdir(output_folder+"models/")
        else:
            print("folder {} exists".format((output_folder)))

# For graph plotting
x_axis, y_axis = [], []
first_plot = True
epoch_list = []
train_loss_list = []
val_loss_list = []

# Write all options/args into file
file = open(output_folder+'args.txt','w')
file.write(str(args))
file.close()

# Write loss value to file
file = open(output_folder+'loss.csv','w')
file.write("X,EPOCH,ITER,LOSS\n")
count = 0


# Load training image
fs = os.listdir(args.dataset)
imagepaths = []
groundtruthpaths = []
for fn in fs:
    base, ext = os.path.splitext(fn)
    if ext == '.jpg' or ext == '.png':
        imagepath = os.path.join(args.dataset,fn)
        imagepaths.append(imagepath)
        groundtruthpath = os.path.join(args.groundtruth,fn)
        groundtruthpaths.append(groundtruthpath)

n_data = len(imagepaths)
print('num traning images:', n_data)
n_iter = int(n_data / batchsize)
print(n_iter, 'iterations,', n_epoch, 'epochs')

# Load validation set
fs = os.listdir(args.validation_set[0])
validation_inputs = []
validation_groundtruths = []
for fn in fs:
    base, ext = os.path.splitext(fn)
    if ext == '.jpg' or ext == '.png':
        imagepath = os.path.join(args.validation_set[0],fn)
        validation_inputs.append(imagepath)
        groundtruthpath = os.path.join(args.validation_set[1],fn)
        validation_groundtruths.append(groundtruthpath)

n_val = len(validation_inputs)
print('num validation images:', n_val)
n_iter_val = int(n_val / batchsize)
print(n_iter_val, 'iterations,', n_epoch, 'epochs')

# Loss network
vgg = VGG()
serializers.load_npz('vgg16.model', vgg)
if args.initmodel:
    print('load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    vgg.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

O = optimizers.Adam(alpha=args.lr)
O.setup(model)
if args.resume:
    print('load optimizer state from', args.resume)
    serializers.load_npz(args.resume, O)


## Change from Image.open to NIP?? for minc
# style = vgg.preprocess(np.asarray(Image.open(args.style_image).convert('RGB').resize((image_size,image_size)), dtype=np.float32))
# style = xp.asarray(style, dtype=xp.float32)
# style_b = xp.zeros((batchsize,) + style.shape, dtype=xp.float32)
# for i in range(batchsize):
#     style_b[i] = style
# feature_s = vgg(Variable(style_b, volatile=True))
# gram_s = [gram_matrix(y) for y in feature_s]

# Check whether to train or run validation 
# Train
if (validate == 0):
    for epoch in range(n_epoch):
        print('epoch', epoch)

        # Training phase
        start = time.time()

        # use to keep loss over one epoch for averaging at the end
        train_loss_epoch = []

        for i in range(n_iter):
            model.zerograds()
            vgg.zerograds()
            # model.cleargrads()
            # L.cleargrads()
            

            indices = range(i * batchsize, (i+1) * batchsize)
            x = xp.zeros((batchsize, 3, image_size, image_size), dtype=xp.float32)
            groundtruth = xp.zeros((batchsize, 3, image_size, image_size), dtype=xp.float32)
            for j in range(batchsize):
                x[j] = load_image(imagepaths[i*batchsize + j], image_size)
                groundtruth[j] = load_image(groundtruthpaths[i*batchsize + j], image_size)



            xc = Variable(x.copy(), volatile=True)
            x = Variable(x)

            y = model(x)

            feature_groudtruth = vgg(groundtruth)
            # feature = vgg(xc)
            feature_hat = vgg(y)

            
            L_feat = lambda_f * F.mean_squared_error(Variable(feature_groudtruth[feature_layer].data), feature_hat[feature_layer]) # compute for only the output of layer conv3_3

            L_style = Variable(xp.zeros((), dtype=np.float32))
            for f, f_hat in zip(feature_groudtruth, feature_hat):
                # L_style += lambda_s * F.mean_squared_error(gram_matrix(f_hat), Variable(g_s.data))
                L_style += lambda_s * F.mean_squared_error(gram_matrix(f_hat), gram_matrix(f))

            L_tv = lambda_tv * total_variation(y)
            L = L_feat + L_style + L_tv #+ L_pixel 
            

            # Write loss to file
            # file.write(str(count)+","+str(epoch)+","+str(i)+","+str(L.data)+"\n")

            print('(epoch {}) batch {}/{}... training loss is...{} '.format(epoch, i, n_iter, L.data/batchsize))

            train_loss_epoch.append(L.data/batchsize)

            L.backward()
            O.update()

            # count += 1
            
            if args.checkpoint > 0 and i % args.checkpoint == 0:
                serializers.save_npz(output_folder+'models/{}_{}_{}.model'.format(output, epoch, i), model)
                serializers.save_npz(output_folder+'models/{}_{}_{}.state'.format(output, epoch, i), O)

        train_loss_list.append(sum(train_loss_epoch)/len(train_loss_epoch))
        print('AVG TRAIN LOSS: ',train_loss_list[epoch])

        end = time.time()
        print("time for train on this epoch: ", time.strftime("%M:%S", time.gmtime(end - start)))






        ####################################
        #
        #
        #   
        # Calculate loss for validation set

        start = time.time()

        # use to keep loss over one epoch for averaging at the end
        val_loss_epoch = []

        for i in range(n_iter_val):

            indices = range(i * batchsize, (i+1) * batchsize)
            x = xp.zeros((batchsize, 3, image_size, image_size), dtype=xp.float32)
            groundtruth = xp.zeros((batchsize, 3, image_size, image_size), dtype=xp.float32)
            for j in range(batchsize):
                x[j] = load_image(validation_inputs[i*batchsize + j], image_size)
                groundtruth[j] = load_image(validation_groundtruths[i*batchsize + j], image_size)


            xc = Variable(x.copy(), volatile=True)
            x = Variable(x)

            y = model(x)

            feature_groudtruth = vgg(groundtruth)
            feature_hat = vgg(y)

            
            L_feat = lambda_f * F.mean_squared_error(Variable(feature_groudtruth[feature_layer].data), feature_hat[feature_layer]) # compute for only the output of layer conv3_3

            L_style = Variable(xp.zeros((), dtype=np.float32))
            for f, f_hat in zip(feature_groudtruth, feature_hat):
                L_style += lambda_s * F.mean_squared_error(gram_matrix(f_hat), gram_matrix(f))

            L_tv = lambda_tv * total_variation(y)
            L = L_feat + L_style + L_tv 

            # print('validation loss: ', L.data)
            # print('validation loss per image: ', L.data/batchsize)
            val_loss_epoch.append(L.data/batchsize)

        val_loss_list.append(sum(val_loss_epoch)/len(val_loss_epoch))
        print('AVG VAL LOSS: ',val_loss_list[epoch])

        end = time.time()
        print("time for validation on this epoch: ", time.strftime("%M:%S", time.gmtime(end - start)))

        # Plot Loss on live graph
        epoch_list.append(epoch)
        if (first_plot):
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss graph')
            plt.plot(epoch_list,train_loss_list)
            plt.plot(epoch_list,val_loss_list)
            plt.legend(['train_loss', 'val_loss'], loc='upper right')
            first_plot = False
        
        # Reallign axis
        plt.gca().lines[0].set_xdata(epoch_list)
        plt.gca().lines[0].set_ydata(train_loss_list)
        plt.gca().lines[1].set_xdata(epoch_list)
        plt.gca().lines[1].set_ydata(val_loss_list)
        plt.gca().relim()
        plt.gca().autoscale_view()
        plt.pause(0.000001);


        # save model for this epoch
        print('save "style.model"')
        serializers.save_npz(output_folder+'models/{}_{}.model'.format(output, epoch), model)
        serializers.save_npz(output_folder+'models/{}_{}.state'.format(output, epoch), O)

    serializers.save_npz(output_folder+'models/{}.model'.format(output), model)
    serializers.save_npz(output_folder+'models/{}.state'.format(output), O)





# Validate
else:
    print("Validating...")
    for epoch in range(n_epoch):
        print('epoch', epoch)
        for i in range(n_iter):
            # model.zerograds()
            # vgg.zerograds()

            indices = range(i * batchsize, (i+1) * batchsize)
            x = xp.zeros((batchsize, 3, image_size, image_size), dtype=xp.float32)
            groundtruth = xp.zeros((batchsize, 3, image_size, image_size), dtype=xp.float32)
            for j in range(batchsize):
                x[j] = load_image(imagepaths[i*batchsize + j], image_size)
                groundtruth[j] = load_image(groundtruthpaths[i*batchsize + j], image_size)



            xc = Variable(x.copy(), volatile=True)
            x = Variable(x)

            y = model(x)

            feature_groudtruth = vgg(groundtruth)

            # feature = vgg(xc)
            feature_hat = vgg(y)
          
            L_feat = lambda_f * F.mean_squared_error(Variable(feature_groudtruth[feature_layer].data), feature_hat[feature_layer]) # compute for only the output of layer conv3_3

            L_style = Variable(xp.zeros((), dtype=np.float32))
            for f, f_hat in zip(feature_groudtruth, feature_hat):
                # L_style += lambda_s * F.mean_squared_error(gram_matrix(f_hat), Variable(g_s.data))
                L_style += lambda_s * F.mean_squared_error(gram_matrix(f_hat), gram_matrix(f))

            # L_tv = lambda_tv * total_variation(y)
            # L = L_feat + L_style + L_tv
            # print("original Lost:" + str(L.data))


            # # NEW LOSS: Difference between real IMG and y for all img in batch
            # lambda_p = 1.0 # To accelerate learning
            # L_pixel = Variable(xp.zeros((), dtype=np.float32))
            # for j in range(batchsize):
            #     # result = np.uint8(img_test[0].transpose((1, 2, 0)))
            #     groundtruth_img = groundtruth[j] 
            #     # output_img = y.data[j] 

            #     # print(groundtruth_img[0,50,50:60])
            #     # print(output_img[0,50,0:50:60])

            #     result_img = result[j]
            #     # print(groundtruth_img.shape)
            #     # print(result_img.shape)
                

            #     # result_img = np.uint8(result_img.transpose((1, 2, 0)))
            #     # imgplot3 = plt.imshow(result_img)
            #     # plt.show()
            #     # quit()

            #     # mean_squared_error = difference in pixel^2 / C*H*w
            #     # L_pixel += lambda_p * F.mean_squared_error(groundtruth_img, output_img)
            #     L_pixel += lambda_p * F.mean_squared_error(groundtruth_img, result_img)

            L_tv = lambda_tv * total_variation(y)
            L = L_feat + L_style + L_tv #+ L_pixel #
            

            # Write loss to file
            file.write(str(count)+","+str(epoch)+","+str(i)+","+str(L.data)+"\n")

            # Plot Loss on live graph
            x_axis.append(count)
            y_axis.append(L.data)

            if (first_plot):
                plt.xlabel('Epoch and Iterations')
                plt.ylabel('Loss')
                plt.title('Loss graph')
                plt.plot(x_axis,y_axis)
                first_plot = False
            
            # Reallign axis
            plt.gca().lines[0].set_xdata(x_axis)
            plt.gca().lines[0].set_ydata(y_axis)
            plt.gca().relim()
            plt.gca().autoscale_view()
            plt.pause(0.000001);

            print('L_feat is...{} L_style is...{} '.format(L_feat.data, L_style.data))
            print('(epoch {}) batch {}/{}... count {} training loss is...{} '.format(epoch, i, n_iter, count, L.data))


            # L.backward()

            # O.update()

            count += 1
            
            # if args.checkpoint > 0 and i % args.checkpoint == 0:
            #     serializers.save_npz(output_folder+'models/{}_{}_{}.model'.format(output, epoch, i), model)
            #     serializers.save_npz(output_folder+'models/{}_{}_{}.state'.format(output, epoch, i), O)

        print('Finish Validation')
    #     serializers.save_npz(output_folder+'models/{}_{}.model'.format(output, epoch), model)
    #     serializers.save_npz(output_folder+'models/{}_{}.state'.format(output, epoch), O)

    # serializers.save_npz(output_folder+'models/{}.model'.format(output), model)
    # serializers.save_npz(output_folder+'models/{}.state'.format(output), O)







##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
### BACK UP FOR SECTION OF CODES
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################

#             serializers.save_npz('models/{}_{}_{}.model'.format(output, epoch, i), model)
#             serializers.save_npz('models/{}_{}_{}.state'.format(output, epoch, i), O)

#     print('save "style.model"')
#     serializers.save_npz('models/{}_{}.model'.format(output, epoch), model)
#     serializers.save_npz('models/{}_{}.state'.format(output, epoch), O)

# serializers.save_npz('models/{}.model'.format(output), model)
# serializers.save_npz('models/{}.state'.format(output), O)





        # print(y.data[0])

        # img_test = y.data
        # imgplot = plt.imshow(img_test[0,0,:,:]) #, cmap='gray')
        # plt.show()

        # img_test = x.data
        # imgplot2 = plt.imshow(img_test[0,0,:,:])
        # plt.show()

        # img_test = x.data
        # result = np.uint8(img_test[0].transpose((1, 2, 0)))
        # imgplot3 = plt.imshow(result)
        # plt.show()

        # print(x.shape)
        # print(y.data)
        # break  


        # TEST MSE
        # black = np.zeros([10,10,3],dtype=xp.float32)

        # white = np.zeros([10,10,3],dtype=xp.float32)
        # white.fill(255)

        # TEST = F.mean_squared_error(black, white)
        # print(TEST.data)

# def mse(imageA, imageB):
#    # the 'Mean Squared Error' between the two images is the
#    # sum of the squared difference between the two images;
#    # NOTE: the two images must have the same dimension
#    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
#    err = err/float(imageA.shape[0] * imageA.shape[1])
#    return err
