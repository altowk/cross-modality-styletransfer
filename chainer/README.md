# Chainer implementation of "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"
Fast artistic style transfer by using feed forward network.

<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/tubingen.jpg" height="200px">

<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/style_1.png" height="200px">
<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/output_1.jpg" height="200px">

<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/style_2.png" height="200px">
<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/output_2.jpg" height="200px">

- input image size: 1024x768
- process time(CPU): 17.78sec (Core i7-5930K)
- process time(GPU): 0.994sec (GPU TitanX)


## Requirement
- [Chainer](https://github.com/pfnet/chainer)
```
$ pip install chainer
$ sudo CUDA_PATH=/usr/local/cuda-7.5 pip install chainer --no-cache-dir -vvvv
```
/home/student/cuda-7.5

## Prerequisite
Download VGG16 model and convert it into smaller file so that we use only the convolutional layers which are 10% of the entire model.
```
sh setup_model.sh
```

## Train
Need to train one image transformation network model per one style target.
According to the paper, the models are trained on the [Microsoft COCO dataset](http://mscoco.org/dataset/#download).
```
python train.py -s <style_image_path> -d <training_dataset_path> -g <use_gpu ? gpu_id : -1>

python train.py -s style.png -d training_data/

python train.py -s sample_images/style_0.jpg -d train2014/ -g 0

python train.py -s sample_images/style_0.jpg -d sample_images/ -g 4

python train.py -s sample_images/style_0.jpg -d training_data/ --batchsize 4 --epoch 1

python train.py -s sample_images/style_0.jpg -d brain_img/ --batchsize 1 --epoch 1

python train.py -s sample_images/style_0.jpg -d test2014/ --batchsize 4 --epoch 1 -r models/TEST.state


python train_unsupervised.py -s sample_images/style_0.jpg -d train_img/ --batchsize 4 --epoch 3

python train_supervised.py -s sample_images/style_0.jpg -d train_img/ --batchsize 4 --epoch 3 --groundtruth ground_truth/
```

## Generate
```
python generate.py <input_image_path> -m <model_path> -o <output_image_path> -g <use_gpu ? gpu_id : -1>
```

This repo has pretrained models as an example.

- example:
```
python generate.py sample_images/tubingen.jpg -m models/composition.model -o sample_images/output.jpg

python generate.py sample_images/tubingen.jpg -m models/style_0 -o sample_images/output_50epoc_9trainimage.jpg
```
or
```
python generate.py sample_images/tubingen.jpg -m models/seurat.model -o sample_images/output.jpg

python generate.py training_data/file.png -m models/style.model -o sample_images/output.jpg

python generate.py sample_images/BRAIN.png -m models/TEST.model -o sample_images/BRAIN_OUT.png
```

## A collection of pre-trained models
Fashizzle Dizzle created pre-trained models collection repository, [chainer-fast-neuralstyle-models](https://github.com/gafr/chainer-fast-neuralstyle-models). You can find a variety of models.

## Difference from paper
- Convolution kernel size 4 instead of 3.
- Training with batchsize(n>=2) causes unstable result.


## Reference
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://arxiv.org/abs/1603.08155)

Codes written in this repository based on following nice works, thanks to the author.

- [chainer-gogh](https://github.com/mattya/chainer-gogh.git) Chainer implementation of neural-style. I heavily referenced it.
- [chainer-cifar10](https://github.com/mitmul/chainer-cifar10) Residual block implementation is referred.
