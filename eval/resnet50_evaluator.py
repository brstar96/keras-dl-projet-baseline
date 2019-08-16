import os
import numpy as np
import pandas as pd

from base.base_evaluater import BaseEvaluate
from generators.resnet50_generator import Resnet50Generator
from utils.get_steps import get_steps


class Resnet50Evaluator(BaseEvaluate):
    def __init__(self, model, weight_path, train_df, test_df, config):
        super().__init__(model, weight_path, train_df, test_df, config)
        
        self.resnet50_generator = Resnet50Generator(self.config)
        self.train_gen = self.resnet50_generator.df_to_generators(
            self.train_df, os.path.join(self.config.data.base_path, self.config.data.train_path), mode='train')
        self.test_gen = self.resnet50_generator.df_to_generators(
            self.test_df, os.path.join(self.config.data.base_path, self.config.data.test_path), mode='test')
        
        self.num_of_classes = self.config.model.class_num

        self.model = self.model.make_model(self.num_of_classes)
        self.model.load_weights(self.weight_path)
        
    def evaluate(self):
        prediction = self.model.predict_generator(
            generator=self.test_gen,
            steps=get_steps(len(self.test_df), self.config.trainer.batch_size),
            verbose=1
        )

        return prediction
    
    def make_labels(self, class_indices):
        labels = dict((v, k) for k, v in class_indices.items())\
        
        return labels
    
    def make_submission(self, submission_path, out_path):
        prediction = self.evaluate()
        predicted_class_indices = np.argmax(prediction, axis=1)
    
        labels = self.make_labels(self.train_gen.class_indices)
    
        predictions = [labels[k] for k in predicted_class_indices]
    
        submission = pd.read_csv(submission_path)
        submission["class"] = predictions
        submission.to_csv(out_path, index=False)
