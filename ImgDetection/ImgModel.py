import os
import numpy as np
from PIL import Image
from paddle.dataset.image import load_image

"""
    conda install paddlepaddle-gpu==2.2.0 cudatoolkit=11.2 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
    
    pip install paddlex==1.3.10
"""

from paddlex.cls import transforms as trsf
import paddlex as pdx
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

"""
    Noting the project tree should be like:
    -base/
    ├─ImgDetection/
    │  └─Classification_config.json/
    │  └─ImgModel.py
    │
    ├─Mydata/
    │  └─1/
    │    └─...
    │  └─...
    │
    ├─output/
    │  └─best_model/
    │  └─test_model/
    │
    ├─test_img_predict/
    │  └─origin/
    │     └─gallery-1.jpg
    │     └─...
    │
    ├─requirements.txt
"""

# Data Argumentation based by paddleX-1.3.11 API (Supporting below version 1.3.11)
train_trsf = trsf.Compose([
    trsf.RandomCrop(crop_size=224), trsf.RandomHorizontalFlip(),
    trsf.RandomCrop(crop_size=224, lower_scale=0.08, lower_ratio=3.0 / 4, upper_ratio=4.0 / 3),
    trsf.RandomHorizontalFlip(prob=0.5),
    trsf.RandomVerticalFlip(prob=0.5),
    trsf.RandomRotate(rotate_range=30, prob=0.5),
    trsf.RandomDistort(brightness_range=0.9, brightness_prob=0.5, contrast_range=0.9, contrast_prob=0.5,
                       saturation_range=0.9, saturation_prob=0.5, hue_range=18, hue_prob=0.5),
    trsf.Normalize()
])

eval_trsf = trsf.Compose([
    trsf.ResizeByShort(short_size=256),
    trsf.CenterCrop(crop_size=224),
    trsf.RandomDistort(brightness_range=0.9, brightness_prob=0.5, contrast_range=0.9, contrast_prob=0.5,
                       saturation_range=0.9, saturation_prob=0.5, hue_range=18, hue_prob=0.5),
    trsf.Normalize()
])


class ModelClass:
    def __init__(self, modelDir: os.path, selfDir: os.path = None):
        self.modelDir = modelDir
        self.model = None
        self.load_model(selfDir)

    def load_model(self, selfDir=None):
        self.model = pdx.load_model(self.modelDir if selfDir is None else selfDir)

    def prediction(self, img_data: os.path or np.ndarray, transforms=eval_trsf):
        # img1 = load_image(img_data)
        # img2 = Image.open(img_data)
        img = img_data if isinstance(img_data, np.ndarray) else cv2.imread(img_data)
        result = self.model.predict(img, transforms=transforms, topk=2)

        # show text result (Not fixed)
        # print(result)

        # show visual result (Not fixed)
        # pdx.det.visualize(img_data, result, threshold=0.6, save_dir='test_img_predict/prediction/')

        returnData = self.fitting(result)
        return returnData

    @staticmethod
    def fitting(result: list):
        data_null = dict()
        if result[0]['score'] > 0.40 and result[1]['score'] > 0.40:
            data_null[result[0]['category_id']] = result[0]['score']
            data_null[result[1]['category_id']] = result[1]['score']
        elif result[0]['score'] >= 0.59:
            data_null[result[0]['category_id']] = result[0]['score']
        elif result[0]['score'] < 0.59:
            return None
        return data_null

    # Need long long long time training with RTX3050 (about 80h), with Tesla V100 (about 5h)
    # Maybe Because paddleX-1.3.10 cannot fit well with 30-Series of NVIDIA
    # You don't need to training this
    def main_training(self):
        os.system("mkdir $ProjectFileDir$/garbage")
        os.system("tar -zxvf ~/data/data77996/Mydata.tar.gz -C $ProjectFileDir$/garbage")

        # Data Parser
        os.system(
            "paddlex --export_inference "
            "--model_dir=output/ResNet50_vd_ssld/best_model "
            "--save_dir=output/best_model "
            "--fixed_input_shape=[720,540]"
        )

        train_dataset = pdx.datasets.ImageNet(
            data_dir='Mydata',
            file_list='Mydata/train.txt',
            label_list='Mydata/labels.txt',
            transforms=train_trsf,
            shuffle=True,
        )
        eval_dataset = pdx.datasets.ImageNet(
            data_dir='Mydata',
            file_list='Mydata/val.txt',
            label_list='Mydata/labels.txt',
            transforms=eval_trsf
        )

        # Training with paddleX ResNet50_vd_ssld_pretrained weights
        model = pdx.cls.ResNet50_vd_ssld(num_classes=len(train_dataset.labels))

        # print("model.model_type")
        model.train(
            num_epochs=60,
            train_dataset=train_dataset,
            train_batch_size=128,
            eval_dataset=eval_dataset,
            learning_rate=0.025,
            lr_decay_epochs=[40, 54],
            warmup_steps=13600,
            save_dir='output/ResNet50_vd_ssld',
            use_vdl=True
        )

        # Save Model
        os.system(
            "paddlex --export_inference "
            "--model_dir=output/ResNet50_vd_ssld/best_model "
            "--save_dir=output/best_model "
            "--fixed_input_shape=[720,540]"
            # add this parameter, output is best_model, otherwise, output is test_model.
        )

        '''
        2021-11-30 07:13:12 [INFO]	[EVAL] Finished, Epoch=60, acc1=0.873951, acc5=0.968115 .
        2021-11-30 07:13:15 [INFO]	Model saved in output/ResNet50_vd_ssld/epoch_60.
        2021-11-30 07:13:15 [INFO]	Current evaluated best model in eval_dataset is epoch_59, acc1=0.8771210143576357
        '''
