from dotmap import DotMap
from pandas import DataFrame


class BaseGenerator:
    def __init__(self, config: DotMap):
        
        self.img_size = tuple(config.model.img_size[:2])
        self.batch_size = config.trainer.batch_size
        self.seed = config.seed
    
    def df_to_generators(self, df: DataFrame, path: str, is_test: bool):
        raise NotImplementedError
