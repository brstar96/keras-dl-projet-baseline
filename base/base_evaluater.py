class BaseEvaluate:
    def __init__(self, model, weight_path, train_df, test_df, config):
        self.model = model
        self.weight_path = weight_path
        self.train_df = train_df
        self.test_df = test_df
        self.config = config
    
    def evaluate(self):
        raise NotImplementedError
