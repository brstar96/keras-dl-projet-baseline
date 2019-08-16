import os

from dotmap import DotMap
from keras.applications import resnet50
from keras.preprocessing.image import ImageDataGenerator
from pandas import DataFrame
from utils.preprocess_function import get_random_eraser

from base.base_generator import BaseGenerator


class Resnet50Generator(BaseGenerator):
    def __init__(self, config: DotMap):
        super().__init__(config)
        self.seed = config.seed
    
    def df_to_generators(self, df: DataFrame, path: str, mode: str):
        y_col = None if mode == 'test' else 'class'
        class_mode = None if mode == 'test' else 'categorical'
        
        preprocessing_function = get_random_eraser(v_l=0, v_h=255) if mode == 'train' else None
        
        datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
        
        generator = datagen.flow_from_dataframe(
            dataframe=df,
            directory=os.path.abspath(path),
            x_col='img_file',
            y_col=y_col,
            target_size=self.img_size,
            color_mode='rgb',
            class_mode=class_mode,
            batch_size=self.batch_size,
            seed=self.seed,
            shuffle=mode != 'test'
        )
        
        return generator
