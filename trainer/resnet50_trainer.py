import os
import pandas as pd
import numpy as np
from dotmap import DotMap

from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

from base.base_model import BaseModel
from base.base_trainer import BaseTrain
from generators.resnet50_generator import Resnet50Generator
from utils.get_steps import get_steps


class Resnet50Trainer(BaseTrain):
    def __init__(self, model: BaseModel, train_df: pd.DataFrame, config: DotMap):
        super().__init__(model, train_df, config)
        self.resnet50_generator = Resnet50Generator(self.config)
        self.num_of_classes = self.config.model.class_num
        
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []

    def make_callbacks(self, patient=6):
        es = EarlyStopping(
            monitor='val_f1_m',
            patience=patient,
            mode='max',
            verbose=1)
        rr = ReduceLROnPlateau(
            monitor='val_f1_m',
            factor=0.5,
            patience=patient / 2,
            min_lr=0.000001,
            verbose=1,
            mode='max')
        ckpt = ModelCheckpoint(
            filepath=os.path.join(
                self.config.callbacks.checkpoint_dir, '%s-{val_loss:.2f}.hdf5' % self.config.exp.name
            ),
            monitor='val_f1_m',
            verbose=1,
            save_best_only=True,
            mode='max'
        )
        tb = TensorBoard(
            log_dir=self.config.callbacks.tensorboard_log_dir,
            write_graph=self.config.callbacks.tensorboard_write_graph,
        )
    
        return [es, rr, ckpt, tb]

    def make_callbacks_for_k_fold(self, current_k, patient=6):
        es = EarlyStopping(
            monitor='val_f1_m',
            patience=patient,
            mode='max',
            verbose=1)
        rr = ReduceLROnPlateau(
            monitor='val_f1_m',
            factor=0.5,
            patience=patient / 2,
            min_lr=0.000001,
            verbose=1,
            mode='max')
        ckpt = ModelCheckpoint(
            filepath=os.path.join(
                self.config.callbacks.checkpoint_dir, '%d-%s-{val_loss:.2f}.hdf5' % (current_k, self.config.exp.name)
            ),
            monitor='val_f1_m',
            verbose=1,
            save_best_only=True,
            mode='max'
        )
        tb = TensorBoard(
            log_dir=self.config.callbacks.tensorboard_log_dir,
            write_graph=self.config.callbacks.tensorboard_write_graph,
        )
    
        return [es, rr, ckpt, tb]
    
    def split_train_val(self, df: pd.DataFrame):
        its = np.arange(df.shape[0])
        train_idx, val_idx = train_test_split(its,
                                              train_size=self.config.trainer.train_ratio,
                                              random_state=self.config.seed)

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        return train_df, val_df
    
    def train(self):
        model = self.model.make_model(self.num_of_classes)
        
        train_df, val_df = self.split_train_val(self.train_df)
        train_gen = self.resnet50_generator.df_to_generators(
            train_df, os.path.join(self.config.data.base_path, self.config.data.train_path), mode='train')
        val_gen = self.resnet50_generator.df_to_generators(
            val_df, os.path.join(self.config.data.base_path, self.config.data.train_path), mode='val')
        
        hist = model.fit_generator(
            generator=train_gen,
            steps_per_epoch=get_steps(len(train_df), self.config.trainer.batch_size),
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            validation_data=val_gen,
            validation_steps=get_steps(len(val_df), self.config.trainer.batch_size),
            callbacks=self.make_callbacks(6)
        )
        
        self.loss.extend(hist.history['loss'])
        self.acc.extend(hist.history['acc'])
        self.val_loss.extend(hist.history['val_loss'])
        self.val_acc.extend(hist.history['val_acc'])

    def train_with_cv(self, k=3):
        skf = StratifiedKFold(n_splits=k, random_state=self.config.seed)
        
        for i, (train_index, valid_index) in \
                enumerate(skf.split(self.train_df['img_file'], self.train_df['class']), start=1):
            train_df = self.train_df.iloc[train_index, :].reset_index()
            valid_df = self.train_df.iloc[valid_index, :].reset_index()
            
            print("========== K Fold Validation step -> {}/{} ==========".format(i, k))

            train_gen = self.resnet50_generator.df_to_generators(
                train_df, os.path.join(self.config.data.base_path, self.config.data.train_path), mode='train')
            val_gen = self.resnet50_generator.df_to_generators(
                valid_df, os.path.join(self.config.data.base_path, self.config.data.train_path), mode='val')
            
            a = next(train_gen)
            
            model = self.model.make_model(self.num_of_classes)
            hist = model.fit_generator(
                generator=train_gen,
                steps_per_epoch=get_steps(len(train_df), self.config.trainer.batch_size),
                epochs=self.config.trainer.num_epochs,
                verbose=self.config.trainer.verbose_training,
                validation_data=val_gen,
                validation_steps=get_steps(len(valid_df), self.config.trainer.batch_size),
                callbacks=self.make_callbacks_for_k_fold(current_k=i, patient=6)
            )

