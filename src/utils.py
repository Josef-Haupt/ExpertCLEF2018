""" Contains helper functions. """
import itertools
import math
import os
import random
import time
import xml.etree.ElementTree as ET
from distutils.util import strtobool
from glob import glob

import keras
import numpy as np


class ValidationError(Exception):
    """ Dummy class for validation errors. """
    pass


def __print__(string: str):
    """ To bypass a bug with python and the win console/ps. *Does not work.* """
    string = string.encode("utf-8").decode("ascii")
    print(string)


def filter_for_filenames(ignore_filenames, path_list) -> list:
    """ quick n dirty """
    return [el for el in path_list if os.path.basename(el)[:os.path.basename(el).rindex('.')] not in ignore_filenames]


def contains(directory: str, count: int, ignore_list: list) -> bool:
    """
    Tests of a *directory* as atleast *count* files after removing all files from the *inore_list*.
    """
    files = glob(os.path.join(directory, '*'))
    jpgs = [el for el in files if el.endswith(".jpg")]
    jpgs = filter_for_filenames(ignore_list, jpgs)
    xmls = [el for el in files if el.endswith(".xml")]
    xmls = filter_for_filenames(ignore_list, xmls)
    return len(list(zip(jpgs, xmls))) >= count


def compare_filename(x) -> bool:
    """ Tests if the base filenames of a tuple of filenames are equal. """
    y = x[1]
    x = x[0]
    x = os.path.basename(x)
    x = x[:x.rindex('.')]
    y = os.path.basename(y)
    y = y[:y.rindex('.')]
    return x == y


def validate_tuple(to_validate) -> bool:
    """ Tests if every tuple of filenames has the same base name. """
    return all(map(compare_filename, to_validate))


def complete_path(path:str, name:str) -> str:
    """
    Checks how many elements of *name[integer]* already exist in *path*, and returns a valid completion.
    """
    existing_dirs = [el for el in os.listdir(path) if os.path.isdir(os.path.join(path, el)) and el.startswith(name)]
    endings = [int(el[len(name):]) for el in existing_dirs if el[len(name):].isdigit()]

    if len(existing_dirs) == 0 or len(endings) == 0:
        return os.path.join(path, name+'1')

    counter = max(endings) + 1
    path = os.path.join(path, name+str(counter))
    return path


def path_to_abs(path:str, relative_to:str) -> str:
    """ Completes the path if it is not already absolute. """
    return os.path.join(relative_to, path) if not os.path.isabs(path) else path


def combine_IDG(possibilities:list, *iters):
    """ Takes a list of *probabilities* and iterators and chains. """
    if len(possibilities) != len(iters):
        raise TypeError('The lenght of probabilities has to be equal to len(iters)')
    if int(sum(possibilities)) != 1:
        raise TypeError('Probabilites have to sum up to 1.')

    pos_arr = [[i]*int(el*100) for i, el in enumerate(possibilities)]
    pos_arr = list(itertools.chain(*pos_arr))
    while True:
        yield next(iters[random.choice(pos_arr)])


def normalisation(x):
    """ Normalises the input array based on the imagenet data set color values. """
    x /= 127.5
    x -= 1.

    x[..., 3] -= 103.939
    x[..., 2] -= 116.779
    x[..., 1] -= 123.68
    return x


def softmax(x):
    """ DEPRECATED: Normalisies the input array. """
    x = [math.exp(i) for i in x]
    x = [round(i/sum(x), 3) for i in x]
    return x


def softmax2(x):
    """ Compute softmax values for each sets of scores in x. """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def find(f:callable, seq) -> list:
    """ Returns a list of all elements for which f(element of seq) is true. """
    result = []
    for item in seq:
        if f(item):
            result.append(item)
    return result


def checkpoint(path:str) -> keras.callbacks.ModelCheckpoint:
    """ Saves the model every epoch. """
    os.makedirs(path, exist_ok=True)
    return keras.callbacks.ModelCheckpoint(
        os.path.join(path, 'checkpoint.h5'),
        monitor='val_acc',
        save_best_only=True,
        period=5,
        verbose=0)


def csv_logger(path:str) -> keras.callbacks.CSVLogger:
    """ Saves the model history inside log.csv. """
    os.makedirs(path, exist_ok=True)
    return keras.callbacks.CSVLogger(os.path.join(path, 'log.csv'))


def plateau() -> keras.callbacks.ReduceLROnPlateau:
    """ Reduzes the learn rate on a plateu. """
    return keras.callbacks.ReduceLROnPlateau(
        monitor='val_acc',
        factor=0.2,
        patience=5,
        cooldown=50,
        verbose=1)


def constant_schedule() -> keras.callbacks.LearningRateScheduler:
    """ This learn rate schedule reduces the learn rate on pre-set epochs. """
    def scheduling(epoch):
        if epoch < 500:
            return 0.01
        elif epoch < 700:
            return 0.001
        return 0.0001
    return keras.callbacks.LearningRateScheduler(scheduling)


def step_decay(max_epochs:int, initial_lr:float = 0.01) -> keras.callbacks.LearningRateScheduler:
    """ The learnrate halfes every max_epochs // 10. """
    def scheduling(epoch):
        drop = 0.5
        epochs_drop = max_epochs/10
        lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lr
    return keras.callbacks.LearningRateScheduler(scheduling)


def try_get(doc:ET, element:str, default=None):
    """ Tries to extract the element from the xml doc and returns the default value if it fails. """
    try:
        return doc.find(element).text.strip()
    except AttributeError:
        return default


def generate_dirname(paths) -> str:
    """ Generates a directory name for the submission, based on the model_paths. """
    result = []
    for path in paths:
        netname = os.path.dirname(path)
        dirname = netname[:3]
        try:
            i = netname.index('_') + 1
            half = netname[i:]
        except ValueError:
            half = ""
            for el in netname[::-1]:
                flag = False
                if el.isnumeric():
                    half += el
                    flag = True
                if not flag:
                    break
            half = half[::-1]
        finally:
            dirname += half
            run = os.path.basename(path)
            result.append(dirname+'-'+run[3:])
    return "_".join(result)


def user_prompt(question:str) -> bool:
    """ Prompt the yes/no-*question* to the user. """
    while True:
        user_input = input(question + " [y/n]: ").lower()
        try:
            result = strtobool(user_input)
            return result
        except ValueError:
            print("Please use y/n or yes/no.\n")


def format_time(ms) -> str:
    """ Formats a ms-time to d:h:min:s:ms:micros. """
    factors = [1000, 1000, 60, 60, 24]
    bez = ['micros', 'millis', 's', 'min', 'h', 'd']
    micro_s = ms * 1000
    result = ""
    for i, _ in enumerate(factors):
        factor = np.prod(factors[:len(factors)-i], dtype=np.int64)
        time_val = micro_s // factor
        micro_s = micro_s % (factor * time_val) if time_val else micro_s
        if time_val > 0:
            result += " {:.0f} {} ".format(time_val,bez[-(i+1)])
    if micro_s > 0:
        result += " {:.0f} {} ".format(micro_s, bez[0])
    return result


def timeit_formated(formatter:callable):
    """ Decorater Factory with custom formatter. The formatter-function gets the time in milliseconds. """
    def _timeit(func:callable):
        """ Decorator that times a function. """
        def wrapper(*args, **kwargs):
            start = time.clock()
            result = func(*args,**kwargs)
            end = time.clock()
            print("{} took {}".format(func.__name__, formatter((end-start)*1000)))
            return result
        return wrapper
    return _timeit


def timeit(func:callable):
    """ Decorator that times a function. """
    def wrapper(*args, **kwargs):
        start = time.clock()
        result = func(*args,**kwargs)
        end = time.clock()
        print("{} took {:.3f} ms".format(func.__name__, (end-start)*1000))
        return result
    return wrapper
