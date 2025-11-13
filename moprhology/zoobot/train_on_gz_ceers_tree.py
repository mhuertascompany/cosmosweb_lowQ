'''
Load the Zoobot model and finetune it on GZ CEERS decision tree. 
'''

import logging
import os

import pandas as pd
import numpy as np
import albumentations as A
from sklearn.model_selection import train_test_split

from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule

from zoobot.pytorch.training import finetune
from zoobot.pytorch.predictions import predict_on_catalog
from zoobot.pytorch.estimators import define_model
from gz_ceers_schema import gz_ceers_schema
from To3d import To3d

# from pytorch_lightning.accelerators.cuda import CUDAAccelerator

# os.environ['CUDA_VISIBLE_DEVICES']="1"

FILTER = 'f444w'


logging.basicConfig(level=logging.INFO)

# checkpoint with pretrained weights, downloaded from: https://zoobot.readthedocs.io/en/latest/data_notes.html
checkpoint_loc = '../checkpoints/effnetb0_greyscale_224px.ckpt'

# directory for saving the finetuned model checkpoint
save_dir = f'/n03data/huertas/CEERS/zoobot/models/finetune_tree_result/{FILTER}'

# self-defined GZ CEERS question tree schema
schema = gz_ceers_schema

accelerator = 'gpu'
devices = 1
batch_size = 64
prog_bar = False
max_galaxies = None

# path for the matched catalog
catalog = pd.read_csv(f"/n03data/huertas/CEERS/zoobot/match_catalog_{FILTER}.csv")

# apply a train-test-valuation ratio of 7:2:1
train_val_catalog, test_catalog = train_test_split(catalog, test_size=0.2)
train_catalog, val_catalog = train_test_split(train_val_catalog, test_size=0.125)

# transform parameters as in original Zoobot
crop_scale_bounds = (0.7, 0.8)
crop_ratio_bounds = (0.9, 1.1)
resize_after_crop = 224     # must match how checkpoint below was trained

# self-defined data module
datamodule = GalaxyDataModule(
    label_cols=schema.label_cols,
    train_catalog=train_catalog,
    val_catalog=val_catalog,
    test_catalog=test_catalog,
    batch_size=batch_size,
    custom_albumentation_transform=A.Compose([
        A.Lambda(image=To3d(),always_apply=True),
        A.Rotate(limit=180, interpolation=1,
            always_apply=True, border_mode=0, value=0),
        A.RandomResizedCrop(
            height=resize_after_crop,  # after crop resize
            width=resize_after_crop,
            scale=crop_scale_bounds,  # crop factor
            ratio=crop_ratio_bounds,  # crop aspect ratio
            interpolation=1,  # This is "INTER_LINEAR" == BILINEAR interpolation.
            always_apply=True
        ),  # new aspect ratio
        A.VerticalFlip(p=0.5),
    ]),  
)

#model = finetune.FinetuneableZoobotTree(
#    zoobot_checkpoint_loc=checkpoint_loc,
#    schema=schema)

model = finetune.FinetuneableZoobotTree(
        name='hf_hub:mwalmsley/zoobot-encoder-efficientnet_b0',
        #name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano',
        schema=schema,
        n_blocks=0
        # n_blocks=5,
        # lr_decay=0.3,
        # learning_rate=5e-6
    )

# print(CUDAAccelerator.is_available())
trainer = finetune.get_trainer(save_dir=save_dir, logger=None, accelerator=accelerator)
trainer.fit(model, datamodule)

# now save predictions on test set to evaluate performance (not necessary if you plan to predict on the full catalog)
trainer_kwargs = {'devices': 1, 'accelerator': accelerator}
predict_on_catalog.predict(
    test_catalog,
    model,
    n_samples=5,
    label_cols=schema.label_cols,
    save_loc=os.path.join(save_dir, f'test_predictions_{FILTER}_effnet_3.1.csv'),
    datamodule_kwargs={
        'custom_albumentation_transform':A.Compose([
            A.Lambda(image=To3d(),always_apply=True),
            A.Rotate(limit=180, interpolation=1,
                always_apply=True, border_mode=0, value=0),
            A.RandomResizedCrop(
                height=resize_after_crop,  # after crop resize
                width=resize_after_crop,
                scale=crop_scale_bounds,  # crop factor
                ratio=crop_ratio_bounds,  # crop aspect ratio
                interpolation=1,  # This is "INTER_LINEAR" == BILINEAR interpolation. 
                always_apply=True
            ),  # new aspect ratio
            A.VerticalFlip(p=0.5),
        ]),
        'batch_size':batch_size
    },
    trainer_kwargs=trainer_kwargs
)