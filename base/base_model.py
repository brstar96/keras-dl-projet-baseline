class BaseModel:
    def __init__(self, config):
        self.config = config
        self.model = None

    def save(self, checkpoint_path):
        """save function that save the checkpoint in the path defined in the config file"""
        if self.model is None:
            raise Exception('You have to build the model first!')

        print('Saving model...')
        self.model.save_weights(checkpoint_path)
        print('Model saved')

    def load(self, checkpoint_path):
        """load latest checkpoint from the experiment path defined in the config file"""
        if self.model is None:
            raise Exception('You have to build the model first!')

        print('Loading model checkpoint {} ...'.format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print('Model loaded')

    def make_model(self, num_of_classes):
        raise NotImplementedError
