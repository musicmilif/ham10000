import os
import logging
from glob import glob
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib.pylab as plt


LOG_FORMAT = '%(asctime)-15s %(levelname)-5s %(name)-15s - %(message)s'


def setup_logging(log_path=None, log_level='DEBUG', logger=None, fmt=LOG_FORMAT):
    """Prepare logging for the provided logger.
    Args:
        log_path (str, optional): full path to the desired log file.
        debug (bool, optional): log in verbose mode or not.
        logger (logging.Logger, optional): logger to setup logging upon,
            if it's None, root logger will be used.
        fmt (str, optional): format for the logging message.
    """
    logger = logger if logger else logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers = []

    fmt = logging.Formatter(fmt=fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.info('Log file is %s', log_path)


def create_loss_plot(exp_dir, epochs, train_losses, test_losses):
    """Plot losses and save.

    Args:
        exp_dir (str): experiment directory.
        epochs (list): list of epochs (x-axis of loss plot).
        train_losses (list): list with train loss during each epoch.
        test_losses (list): list with test loss during each epoch.

    """
    f = plt.figure()
    plt.title("Loss plot")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.plot(epochs, train_losses, 'b', marker='o', label='train loss')
    plt.plot(epochs, test_losses, 'r', marker='o', label='test loss')
    plt.legend()
    plt.savefig(os.path.join(exp_dir, 'loss.png'))


def generate_meta(data_dir):
    """Generate meta dataframe.
    """
    img_path_mapping = {
        os.path.basename(path).split('.')[0]: path
        for path in glob(os.path.join(data_dir, f'images/*/*.jpg'))
    }    
    
    df = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))
    df['path'] = df['image_id'].map(img_path_mapping)
    df['dx'] = df['dx'].astype('category')
    df['target'] = df['dx'].cat.codes

    return df


def stratified_split(df: pd.DataFrame, train_frac: float=.7, seed=2019):
    """Using stratified split to generate training, validation and testing ids.
    Arg:
        df (pd.DataFrame): a pandas dataframe with ['image_id', 'target'] columns.
        train_frac (float): fraction of data to use for training.
    Return:
        train_ids (np.ndarray): ids for training.
        valid_ids (np.ndarray): ids for validation.
        test_ids (np.ndarray): ids for testing.
    """
    sss = StratifiedShuffleSplit(train_size=train_frac, random_state=seed)
    ids = df['image_id'].values
    train_idx, test_idx = next(sss.split(ids, df['target'].values))
    train_ids, test_ids = ids[train_idx], ids[test_idx]

    return train_ids, test_ids
