class BaseTrain:
    def __init__(self, model, train_df, config):
        self.model = model
        self.train_df = train_df
        self.config = config
        
    def train(self):
        raise NotImplementedError
