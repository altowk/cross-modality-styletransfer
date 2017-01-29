#!/bin/sh

# F 1.0 , S 5.0
python train_supervised.py -s t1/train/2216.mnc.png -d t1/train/ --groundtruth t2/train/ --batchsize 11 --epoch 50 -g 0 --folder model_2216_f1_s5/ --lambda_feat 1.0  --lambda_style 5.0

python generate.py t1/validate/6850.mnc.png -m model_2216_f1_s5/models/2216.mnc.model -o model_2216_f1_s5/2216_50_MNC.png

# F 1.0 , S 3.0
python train_supervised.py -s t1/train/2216.mnc.png -d t1/train/ --groundtruth t2/train/ --batchsize 11 --epoch 50 -g 0 --folder model_2216_f1_s3/ --lambda_feat 1.0  --lambda_style 3.0

python generate.py t1/validate/6850.mnc.png -m model_2216_f1_s3/models/2216.mnc.model -o model_2216_f1_s3/2216_50_MNC.png

# F 3.0, S 2.0
python train_supervised.py -s t1/train/2216.mnc.png -d t1/train/ --groundtruth t2/train/ --batchsize 11 --epoch 50 -g 0 --folder model_2216_f3_s2/ --lambda_feat 1.0  --lambda_style 3.0

python generate.py t1/validate/6850.mnc.png -m model_2216_f3_s2/models/2216.mnc.model -o model_2216_f3_s2/2216_50_MNC.png



python train_supervised_CHECKCHAINER.py -d t1/ --groundtruth t2/ --batchsize 1 --epoch 1 --folder validate_model_2216_f1_s5/ --lambda_feat 1.0  --lambda_style 3.0 --validate 1 -r new_model/f1_s3/2216.mnc.state -i new_model/f1_s3/2216.mnc_10.model





python train_supervised_CHECKCHAINER.py -d t1/ --groundtruth t2/ --batchsize 1 --epoch 3 -g -1 --lambda_feat 1.0  --lambda_style 3.0 -r new_model/f1_s3/2216.mnc.state -i new_model/f1_s3/2216.mnc.model

python train_supervised_CHECKCHAINER.py -d t1/ --groundtruth t2/ --batchsize 1 --epoch 30 -g -1 --lambda_feat 1.0  --lambda_style 3.0 -r model_f1.0_s3.0_tv1e-06_lf2/models/f1.0_s3.0_tv1e-06_lf2.state -i model_f1.0_s3.0_tv1e-06_lf2/models/f1.0_s3.0_tv1e-06_lf2.model --validation_set t1_val/ t2_val/

python train_supervised.py -s t1/486.mnc.png -d t1/ --groundtruth t2/ --batchsize 1 --epoch 3 --folder model_2216_f1_s3/ --lambda_feat 1.0  --lambda_style 3.0 -r new_model/f1_s3/2216.mnc.state -i new_model/f1_s3/2216.mnc.model




python generate.py t1/486.mnc.png -m model_f1.0_s3.0_tv1e-06_lf2/models/f1.0_s3.0_tv1e-06_lf2.model -o model_f1.0_s3.0_tv1e-06_lf2/486_GEN_MNC.png