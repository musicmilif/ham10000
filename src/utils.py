import os
import logging
import itertools
from glob import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib.pylab as plt
import seaborn as sns


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
    plt.close()


def create_confusion_matrix(exp_dir, cm, classes, normalize=False):
    """Plot confusion matrix on given dataset.
    Args:
        exp_dir (str): experiment folder directory path.
        cm (np.ndarray): confusion matrix.
        classes (list): list of name for classes.
        normalize (bool): normalize the count into percentage.
    """
    vmin, vmax = cm.min(), cm.max()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, 2)
        vmin, vmax = 0, 1

    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, cmap="Blues", vmin=vmin, vmax=vmax, square=True, 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion matrix', fontdict={'fontsize': 20})
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'confusion_mtx.png'), dpi=120)
    plt.close()


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


def over_sample(df: pd.DataFrame, frac=.3, imbalance=True):
    """If the distribution of target is imbalanced, do oversample
    Args:
        df (pd.DataFrame): training dataframe.
        frac (float): between 0 and 1, to reduce some data.
        imbalance (bool): to do oversampling or not.
    """
    if imbalance:
        weights = df['target'].value_counts()
        weights = ((2*weights.max())/(3*weights)).apply(int).sort_index().to_list()

        for cat in range(len(weights)):
            df = pd.concat([df]+[df.loc[df['target']==cat]]*weights[cat], axis=0)
        df = df.sample(frac=frac).reset_index(drop=True)

    return df