"""
Contains all configurations regarding model, train and validation directories.
"""
import argparse
import os
import random
import shutil
import warnings
import xml.dom.minidom
import xml.etree.ElementTree as ET
from contextlib import suppress
from glob import glob

import pandas as pd
from PIL import Image

import utils

try:
    doc = ET.parse('config.xml')
    CLEAN_DATA_SETS = [el.attrib['path'] for el in doc.find('datasets/clean')]
    NOISY_DATA_SETS = [el.attrib['path'] for el in doc.find('datasets/noisy')]

    with suppress(AttributeError):
        COMBINED_TRAIN_SET = doc.find('datasets/all').attrib['path']
    MAIN_DATASET_PATH = CLEAN_DATA_SETS[0]

    VALIDATION_PATH = doc.find('validation').attrib['path']
    TESTSET_PATH = doc.find('testset').attrib['path']

    with suppress(KeyError):
        PC_NAME = doc.find('.').attrib['for']

    MODELS_PATH = doc.find('models').attrib['path']
except FileNotFoundError:
    warnings.warn("There is no config.xml, please use -c to create one.")

with open('ignore.txt', 'r') as ignore_file:
    IGNORE_LIST = ignore_file.read().splitlines()

try:
    config = xml.dom.minidom.parse("testing.xml")
    TEST_DATASET_PATH = [el for el in config.getElementsByTagName('dataset') if el.getAttribute('name') == 'PlantCLEF2017Train1EOL'][0].getAttribute('path')
    TEST_VALIDATION_PATH = config.getElementsByTagName('validation')[0].getAttribute('path')
    try:
        TEST_MODELS_PATH = config.getElementsByTagName('models')[0].getAttribute('path')
    except IndexError:
        TEST_MODELS_PATH = None
    try:
        TEST_TEST_SET_PATH = config.getElementsByTagName('testset')[0].getAttribute('path')
    except IndexError:
        TEST_TEST_SET_PATH = None
    del config
except Exception:
    warnings.warn("There is an error inside your testing.xml.")


@utils.timeit_formated(utils.format_time)
def split_val_v2(val_path:str, split_percentage_per_class:float, _set:str = 'clean', _max=None):
    """
    Takes the given percentage of images from each class und pastes them into the validation directory.
    If a class contains only one sample, the samples is only copied and will be inside train and validation set.
    """
    dirs = glob(os.path.join(MAIN_DATASET_PATH, '*')) if _set == 'clean' else glob(os.path.join(NOISY_DATA_SETS[0], '*'))
    for _dir in dirs:
        files = glob(os.path.join(_dir, '*'))
        jpgs = [el for el in files if el.endswith(".jpg")]
        jpgs = utils.filter_for_filenames(IGNORE_LIST, jpgs)

        xmls = [el for el in files if el.endswith(".xml")]
        xmls = utils.filter_for_filenames(IGNORE_LIST, xmls)

        jpgs = sorted(jpgs, key=lambda el: int(os.path.basename(el)[:os.path.basename(el).rindex('.')]))
        xmls = sorted(xmls, key=lambda el: int(os.path.basename(el)[:os.path.basename(el).rindex('.')]))

        sample_label = list(zip(jpgs, xmls))
        if not utils.validate_tuple(sample_label):
            raise utils.ValidationError("Not all elements have a label/description.")
        valid_count = int(split_percentage_per_class*len(sample_label))

        if _max is not None:
            valid_count = _max if valid_count > _max else valid_count

        if valid_count:
            while valid_count > 0:
                try:
                    to_move = random.choice(sample_label)
                except IndexError:
                    break

                classid = os.path.basename(os.path.dirname(to_move[0]))
                if utils.contains(os.path.dirname(to_move[0]), 2, IGNORE_LIST):
                    parent = os.path.join(val_path, classid)
                    os.makedirs(parent, exist_ok=True)
                    shutil.move(to_move[0], os.path.join(parent, os.path.basename(to_move[0])))
                    shutil.move(to_move[1], os.path.join(parent, os.path.basename(to_move[1])))
                    sample_label.remove(to_move)
                    valid_count -= 1
        else:
            if len(sample_label) > 1:
                try:
                    to_move = random.choice(sample_label)
                except IndexError:
                    break

                classid = os.path.basename(os.path.dirname(to_move[0]))
                parent = os.path.join(val_path, classid)
                os.makedirs(parent)
                shutil.move(to_move[0], os.path.join(parent, os.path.basename(to_move[0])))
                shutil.move(to_move[1], os.path.join(parent, os.path.basename(to_move[1])))
            else:
                to_move = sample_label[0]
                classid = os.path.basename(os.path.dirname(to_move[0]))
                parent = os.path.join(val_path, classid)
                os.makedirs(parent)
                shutil.copy(to_move[0], os.path.join(parent, os.path.basename(to_move[0])))
                shutil.copy(to_move[1], os.path.join(parent, os.path.basename(to_move[1])))


def percentage(value) -> float:
    """ Tests if a float is between 0. and 1. """
    try:
        val = float(value)
    except TypeError:
        raise argparse.ArgumentTypeError('-s', 'Value has to be between 0 and 1.')

    if val > 1.0 or val < 0.0:
        raise argparse.ArgumentTypeError('-s', 'Value has to be between 0 and 1.')

    return val


def path(string:str) -> str:
    """ Creates the fiven path tree. """
    os.makedirs(string, exist_ok=True)
    return string


def create_config(paths:list):
    """ Creates a configuration file based on the given paths. """
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<config>')
    lines.append('<datasets>')
    lines.append('<clean>')
    lines.append('<dataset path="{}" />'.format(paths[0]))
    lines.append('</clean>')
    lines.append('<noisy>')
    lines.append('</noisy>')
    lines.append('</datasets>')
    lines.append('<validation path="{}" />'.format(paths[1]))
    lines.append('<testset path="{}" />'.format(paths[3]))
    lines.append('<models path="{}" />'.format(paths[2]))
    lines.append('</config>')
    with open('config.xml', 'w') as config_file:
        config_file.write("\n".join(lines))


@utils.timeit_formated(utils.format_time)
def split_test_set(classes:int, val_samples:int, min_files:int):
    """ Splits a mini set from the main data set for testing pruposes. """
    dirs = glob(os.path.join(MAIN_DATASET_PATH, '*'))
    dirs = [el for el in dirs if len(glob(os.path.join(el, '*.jpg'))) > min_files]

    test_dir = os.path.join(os.path.dirname(MAIN_DATASET_PATH), 'testing')
    os.makedirs(test_dir)

    for _ in range(classes):
        try:
            el = random.choice(dirs)
        except IndexError:
            break
        dirs.remove(el)
        files = glob(os.path.join(el, '*'))
        files = utils.filter_for_filenames(IGNORE_LIST, files)

        train_dir = os.path.join(test_dir, 'data', os.path.basename(el))
        valid_dir = os.path.join(test_dir, 'valid', os.path.basename(el))

        os.makedirs(train_dir)
        os.makedirs(valid_dir)

        jpgs = [el for el in files if el.endswith(".jpg")]
        xmls = [el for el in files if el.endswith(".xml")]

        jpgs = sorted(jpgs, key=lambda el: int(os.path.basename(el)[:os.path.basename(el).rindex('.')]))
        xmls = sorted(xmls, key=lambda el: int(os.path.basename(el)[:os.path.basename(el).rindex('.')]))

        combined = list(zip(jpgs,xmls))
        for _ in range(val_samples):
            val_sample = random.choice(combined)
            combined.remove(val_sample)
            shutil.copy(val_sample[0], valid_dir)
            shutil.copy(val_sample[1], valid_dir)

        for train_samples in combined:
            shutil.copy(train_samples[0], train_dir)
            shutil.copy(train_samples[1], train_dir)


@utils.timeit_formated(utils.format_time)
def unify_val_n_train_sets(val_path:str, train_path:str):
    """ Moves all files from the validation set back into the main data set. """
    files = glob(os.path.join(val_path,"**"), recursive=True)

    jpgs = [el for el in files if el.endswith(".jpg")]
    xmls = [el for el in files if el.endswith(".xml")]

    jpgs = sorted(jpgs, key=lambda el: int(os.path.basename(el)[:os.path.basename(el).rindex('.')]))
    xmls = sorted(xmls, key=lambda el: int(os.path.basename(el)[:os.path.basename(el).rindex('.')]))

    combined = list(zip(jpgs, xmls))
    utils.validate_tuple(combined)

    for img, _xml in combined:
        with suppress(shutil.Error):
            shutil.move(img, os.path.join(train_path, os.path.basename(os.path.dirname(img))))
            shutil.move(_xml, os.path.join(train_path, os.path.basename(os.path.dirname(_xml))))
    shutil.rmtree(val_path)


@utils.timeit_formated(utils.format_time)
def rename_invalid(_path:str):
    """ Renames all '*.jpg' which are invalid to '*.invalid'. """
    dirs = glob(os.path.join(_path, '*'))
    for _dir in dirs:
        files = glob(os.path.join(_dir, '*.jpg'))
        for file in files:
            size = os.path.getsize(file)
            if size == 37:
                os.rename(file, file.replace('.jpg', '.invalid'))


def check_all_files(_path:str):
    """
    Checks if all JPG-files inside a path can be read by PIL.
    If not they are renamed to "*.invalid".
    """
    dirs = glob(os.path.join(_path, '*'))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _dir in dirs:
            files = glob(os.path.join(_dir, '*.jpg'))
            for file in files:
                try:
                    with Image.open(file) as im:
                        im.load()
                except Exception:   # The type if exception/warning does not matter.
                                    # possible are: (OSError, IOError, ValueError, UnboundLocalError)
                                    # and EXIF warnings which can lead to exceptions while training.
                    print("{} renamed.".format(file))
                    os.rename(file, file.replace('.jpg', '.invalid'))


@utils.timeit_formated(utils.format_time)
def invalid_to_jpg(_path:str):
    """ Renames all '.invalid' files back to '*.jpg'. """
    files = glob(os.path.join(_path, '**', '*.invalid'))
    for file in files:
        os.rename(file, file.replace('.invalid', '.jpg'))


@utils.timeit_formated(utils.format_time)
def copy_files(target:str, *args):
    """
    Copies all files in the direcotries in args into the target direcotry.
    Files with the same parent directory get put into the same directory.
    """
    for arg in args:
        dirs = glob(os.path.join(arg, "*"))
        for _dir in dirs:
            files = glob(os.path.join(_dir, "*"))
            dst = os.path.join(target, os.path.basename(_dir))
            with suppress(FileExistsError):
                os.mkdir(dst)
            for file in files:
                shutil.copy(file, dst)


@utils.timeit_formated(utils.format_time)
def clean(max_epochs:int):
    """ Deletes all runs which only lasted for *max_epochs*. """
    relevant_files = glob(os.path.join(MODELS_PATH, "**", "log.csv"), recursive=True)
    if not TEST_MODELS_PATH.startswith(MODELS_PATH):
        relevant_files += glob(os.path.join(TEST_MODELS_PATH, "**", "log.csv"), recursive=True)

    for log_file in relevant_files:
        try:
            log = pd.read_csv(log_file)
            if log.shape[0] < max_epochs:
                shutil.rmtree(os.path.dirname(log_file))
        except pd.io.common.EmptyDataError:
            shutil.rmtree(os.path.dirname(log_file))


def handle_args(**kwargs):
    """ Executes all commands. """
    if kwargs['create_config']:
        print("A config file is beeing created.")
        create_config(kwargs['create_config'])

    if kwargs['split_validation']:
        print("This operation may take some time.")
        split_val_v2(VALIDATION_PATH, kwargs['split_validation'])
    elif kwargs['split_validation_max']:
        split_val_v2(
            VALIDATION_PATH,
            kwargs['split_validation'][0],
            _set=kwargs['split_validation_max'][1],
            _max=kwargs['split_validation_max'][2])

    if kwargs['unify_val_n_train_sets']:
        print("Files from the training set are moved to the validation set.")
        unify_val_n_train_sets(VALIDATION_PATH, MAIN_DATASET_PATH)

    if kwargs['split_testing']:
        print("A small test set is beeing created.")
        classes = kwargs['split_testing'][0]
        val_count = kwargs['split_testing'][1]
        min_files = kwargs['split_testing'][2]
        split_test_set(classes, val_count, min_files)

    if kwargs['invald_jpg']:
        print("'*.invalid' files are renamed to '*.jpg'.")
        invalid_to_jpg(kwargs['invald_jpg'])

    if kwargs['rename_invalid']:
        print("All invalid files are being renames to '*.invalid'.")
        print("This may take a while.")
        rename_invalid(kwargs['rename_invalid'])
        check_all_files(kwargs['rename_invalid'])

    if kwargs['combine_clean_noisy']:
        print("Data sets are combined inside {}.".format(kwargs['combine_clean_noisy']))
        copy_files(kwargs['combine_clean_noisy'], NOISY_DATA_SETS[0], MAIN_DATASET_PATH)

    if kwargs['clean']:
        if utils.user_prompt("Do you really want to delete all runs with 10 or less Epochs (also in the test dir)?"):
            print("Cleaning up...")
            clean(max_epochs=kwargs['clean'])


def main():
    """ Einstiegspunkt """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t',
        '--split_testing',
        nargs=3,
        type=int,
        metavar=('CLASS_COUNT','VAL_COUNT','MIN_FILES'),
        help='Test set will contain *CLASS_COUNT* classes with min. *MIN_FILES* files, *VAL_COUNT* images will be used for validation.')
    parser.add_argument(
        '-c',
        '--create_config',
        type=path,
        nargs=4,
        metavar=('train_path','val_path','models_path','test_path'),
        help='Value 1 is the path to the training set, value 2 validation set path and the third the path to the model directory.')
    parser.add_argument(
        '-s',
        '--split_validation',
        type=percentage,
        help='Moves part of the main data set into a validation set.')
    parser.add_argument(
        '--split_validation_max',
        metavar=('PERCENTATGE','SET','MAX_SAMPLES'),
        nargs=3,
        help='Moves part of the main data set into a validation set but limits the max number of files.')
    parser.add_argument(
        '-u',
        '--unify_val_n_train_sets',
        action='store_true',
        help='Moves all files from the validation set back into the training set.')
    parser.add_argument(
        '-r',
        '--rename_invalid',
        type=str,
        help="Renames all invalid files in the noisy data set to '*.invalid'.")
    parser.add_argument(
        '--combine_clean_noisy',
        type=str,
        help="Copies all files from the clean and the noisy data set into a seperate dictioniary.")
    parser.add_argument(
        '--clean',
        type=int,
        metavar='MAX_EPOCHS',
        help="Cleans the model directories of all models with less then *MAX_EPOCHS* epochs.")
    parser.add_argument(
        '--invalid_jpg',
        type=str,
        help="Renames all invalid files back to '.jpg'.")

    args = parser.parse_args()
    handle_args(**vars(args))


if __name__ == '__main__':
    main()
