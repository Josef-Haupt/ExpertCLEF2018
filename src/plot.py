""" This module contains functions for plotting. """
import argparse
import itertools
import os
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import pandas as pd

import config
import utils


def plot(*paths:str):
    """ Plots the history of all paths in the same figure. """
    f_val_acc = plt.figure()
    f_val_acc.suptitle('Validation Accuracy')
    ax_val_acc = f_val_acc.add_subplot(111)
    ax_val_acc.set_xlabel('Epochs')
    ax_val_acc.set_ylabel('Error')

    f_val_top3acc = plt.figure()
    f_val_top3acc.suptitle('Validation Top3-Accuracy')
    ax_val_top3acc = f_val_top3acc.add_subplot(111)
    ax_val_top3acc.set_xlabel('Epochs')
    ax_val_top3acc.set_ylabel('Error')

    for path in paths:
        frame = pd.read_csv(os.path.join(path, 'history.csv'))
        ax_val_acc.plot(frame['val_acc'])
        ax_val_top3acc.plot(frame['val_top3_acc'])

    ax_val_top3acc.legend([os.path.basename(el) for el in paths], loc='upper left')
    ax_val_acc.legend([os.path.basename(el) for el in paths], loc='upper left')
    plt.show()


def get_optimizer_data(doc) -> dict:
    """ Gets all the relevant information about the used optimizer from the config file. """
    optim_data = {}
    optim_data.update({'nesterov': utils.try_get(doc, './optimizer/nesterov')})
    optim_data.update({'learn_rate': utils.try_get(doc, './optimizer/learn_rate')})
    optim_data.update({'schedule': utils.try_get(doc, './optimizer/schedule')})
    optim_data.update({'decay': utils.try_get(doc, './optimizer/decay')})
    optim_data.update({'momentum': utils.try_get(doc, './optimizer/momentum')})
    optim_data = {key:value for key,value in optim_data.items() if value is not None}
    return optim_data


def get_str_from_dic(dic:dict) -> list:
    """ Converts a dictionry in a formatted string. """
    string = []
    for key, item in dic.items():
        part = '{:<15}{:>11} '.format(key+':',item)
        if len(string)%2 == 0:
            part+='|'
        string.append(part)
    if string:
        left = [el for i,el in enumerate(string) if not i % 2]
        right = [el for i,el in enumerate(string) if i % 2]
        zipped = itertools.zip_longest(left,right, fillvalue='')
        string = [a+' '+b for a,b in list(zipped)]
    return string


def get_augment_data(doc:ET) -> dict:
    """ Gets all the information about the used data augmentation from the config. """
    augment_data = {}
    augment_data.update({'fill_mode': utils.try_get(doc, './optimizer/fill_mode')})
    augment_data.update({'horizontal_flip': utils.try_get(doc, './optimizer/horizontal_flip')})
    augment_data.update({'vertical_flip': utils.try_get(doc, './optimizer/vertical_flip')})
    augment_data.update({'width_shift_range': utils.try_get(doc, './optimizer/width_shift_range')})
    augment_data.update({'height_shift_range': utils.try_get(doc, './optimizer/height_shift_range')})
    augment_data.update({'shear_range': utils.try_get(doc, './optimizer/shear_range')})
    augment_data.update({'rotation_range': utils.try_get(doc, './optimizer/rotation_range')})
    augment_data.update({'zoom_range': utils.try_get(doc, './optimizer/zoom_range')})
    augment_data = {key:value for key,value in augment_data.items() if value is not None}
    return augment_data


def print_info(*paths:str):
    """ Prints all informations for every path. """
    for path in paths:
        doc =  ET.parse(os.path.join(path, 'model_config.xml'))
        model_name = doc.find('.').attrib['title']
        opt_name = doc.find('./optimizer/name').text.strip()
        epochs = doc.find('./epochs').text.strip()
        tm = doc.find('./train_mode').text.strip()
        samples = doc.find('./samples_per_epoch').text.strip()
        batch_size = doc.find('./batch_size').text.strip()
        print('{:*^54}'.format(' ' + os.path.basename(path) + ' - ' + model_name + ' '))
        print('{:<15}{:>11} | {:<15}{:>11}'.format('epochs: ', epochs, 'samples_epoch: ', samples))
        print('{:<15}{:>11} | {:<15}{:>11}'.format('batch_size: ', batch_size, 'train_mode: ', tm))
        print('{:<15}{:>11} | {:-<26}'.format('optimizer: ', opt_name, ''))
        optim_data = get_optimizer_data(doc)
        optim_string = get_str_from_dic(optim_data)
        if optim_string:
            print('\n'.join(optim_string))
        print('{:<15}{:>11} | {:-<26}'.format('augmentations: ', '', ''))
        augment_data = get_augment_data(doc)
        augment_string = get_str_from_dic(augment_data)
        if augment_string:
            print('\n'.join(augment_string))


def handle_args(**kwargs):
    """ Handles the command line arguments. """
    rel_to = config.TEST_MODELS_PATH if kwargs['test_mode'] else config.MODELS_PATH
    paths = [utils.path_to_abs(el, rel_to) for el in kwargs['paths']]
    if kwargs['info']:
        print_info(*paths)
    plot(*paths)


def main():
    """ Entry point """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'paths',
        metavar='PATH',
        type=str,
        nargs='+',
        help='Has to be relativ to the models directory inside the config.')
    parser.add_argument(
        '-t',
        '--test_mode',
        action='store_true',
        help="The test path inside testing.xml will be used.")
    parser.add_argument(
        '-i',
        '--info',
        action='store_true',
        help=print_info.__doc__)

    args = parser.parse_args()
    handle_args(**vars(args))


if __name__ == '__main__':
    main()
