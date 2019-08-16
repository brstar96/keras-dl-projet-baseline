import os

import cv2
from dotmap import DotMap
from tqdm import tqdm

from utils import get_df


def crop_images(config: DotMap):
    train_path: str = config.data.train_path
    train_crop_path = train_path + '_crop'
    
    test_path: str = config.data.test_path
    test_crop_path = test_path + '_crop'
    
    train_df, test_df = get_df.get_df_from_config(config)
    
    img_size = tuple(config.model.img_size[:2])
    
    _crop_images(train_path, train_crop_path, train_df, img_size)
    _crop_images(test_path, test_crop_path, test_df, img_size)


def _crop_images(img_path, crop_path, df, img_size):
    if not os.path.isdir(crop_path):
        os.mkdir(crop_path)
    
    for i, row in tqdm(df.iterrows()):
        cropped = _crop_boxing_img(row['img_file'], img_path, df, img_size)
        if not os.path.isfile(os.path.join(crop_path, row['img_file'])):
            cv2.imwrite(os.path.join(crop_path, row['img_file']), cropped)


def _crop_boxing_img(img_name, img_path, df, img_size, margin=5):
    img = cv2.imread(os.path.join(img_path, img_name))
    pos = df.loc[df["img_file"] == img_name, ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].values.reshape(-1)

    height, width = img.shape[:2]
    x1 = max(0, pos[0] - margin)
    y1 = max(0, pos[1] - margin)
    x2 = min(pos[2] + margin, width)
    y2 = min(pos[3] + margin, height)

    return cv2.resize(img[y1:y2, x1:x2], img_size)
