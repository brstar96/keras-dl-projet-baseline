from dotmap import DotMap
from keras import Sequential, layers, optimizers
from keras.layers import GlobalAveragePooling2D
from base.base_model import BaseModel
from keras.applications.resnet50 import ResNet50
from utils.metrics import f1_m, precision_m, recall_m


class Resnet50Model(BaseModel):
    def __init__(self, config: DotMap):
        super().__init__(config)
    
    def make_model(self, num_of_classes):
        input_shape = tuple(self.config.model.img_size)
        output_classes = num_of_classes
        dense_size = 2048
        dropout_ratio = 0.15
        
        model = Sequential()
        
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(layers.Dense(dense_size, activation='relu', kernel_initializer='he_normal'))
        model.add(layers.Dropout(dropout_ratio))
        model.add(layers.Dense(output_classes, activation='softmax', kernel_initializer='he_normal'))
        
        optimizer = optimizers.Adam(lr=self.config.model.learning_rate)
        
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['acc', f1_m, precision_m, recall_m])
        
        return model
