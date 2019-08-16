import os

from model.resnet50_model import Resnet50Model
from trainer.resnet50_trainer import Resnet50Trainer
from eval.resnet50_evaluator import Resnet50Evaluator
from utils.args import get_args
from utils.config import process_config
from utils.dirs import create_dirs
from utils.get_df import get_df_from_config


def main():
    """
    capture the config path from the run arguments
    then processw the json configuration file
    """
    try:
        args = get_args()
        config = process_config(args.config)
    except FileNotFoundError as e:
        print('missing or invalid arguments')
        exit(0)
        return
    
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])
    
    print('Get dataframe from csv')
    train_df, test_df = get_df_from_config(config)
    train_df['class'] = train_df['class'].astype('str')
    train_df = train_df[['img_file', 'class']]
    test_df = test_df[['img_file']]
    
    print('Create model')
    model = Resnet50Model(config)
    
    print('Create trainer')
    trainer = Resnet50Trainer(model, train_df, config)
    
    print('Start training model')
    trainer.train_with_cv(k=4)
    
    # print('Create evaluator')
    # weight_path = 'experiments/2019-08-15/resnet50_k-fold/checkpoints/2-resnet50_k-fold-0.59.hdf5'
    # evaluator = Resnet50Evaluator(model, weight_path, train_df, test_df, config)
    #
    # print('make submission')
    # out_path = os.path.join(config.data.base_path, 'submission_{}.csv'.format(config.exp.name))
    # evaluator.make_submission(os.path.join(config.data.base_path, config.data.submission), out_path)

    
if __name__ == '__main__':
    main()
