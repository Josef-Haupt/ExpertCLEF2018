""" This module contains only analytic functions. """
import argparse
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import config
import utils
from utils import __print__


def invalid_count_all():
    """ Counts the invalid files inside every data set. """
    for path in config.CLEAN_DATA_SETS+config.NOISY_DATA_SETS:
        invalid_count(path)


def invalid_count(path:str):
    """ Prints the invalid file count in a directory. """
    count = len(glob(os.path.join(path, "**", "*.invalid")))
    print("{} Images are invalid in {}.".format(str(count), path))


def validate(_raise:bool = True, validate_class_path:bool = False, _print:bool = False) -> tuple:
    """
    Tests if every train image has a xml description.
    It will also check if every image of one class is on the same folder.
    """
    count_ims, count_xmls = image_count(_print=False)
    fail_paths = []
    wrong_dir = []

    if count_ims != count_xmls:
        folders = glob(os.path.join(config.MAIN_DATASET_PATH, "*"))

    for directory in folders:
        elements = glob(os.path.join(directory,"*"))
        xmls = [el for el in elements if el.endswith(".xml")]
        jpgs = [el for el in elements if el.endswith(".jpg")]
        fail_paths.extend(validate_parts(xmls, ".jpg", _raise=_raise))
        fail_paths.extend(validate_parts(jpgs, ".xml", _raise=_raise))

        if validate_class_path:
            try:
                for _xml in xmls:
                    doc = xml.dom.minidom.parse(_xml)
                    classid = doc.getElementsByTagName('ClassId')[0].childNodes[0].data
                    dirname = os.path.basename(os.path.dirname(_xml))
                    if int(classid) != int(dirname):
                        wrong_dir.append(_xml)
            except xml.parsers.expat.ExpatError:
                pass

    if _print:
        print("File count with no counterpart: {}".format(len(fail_paths)))
        print("Files in the wrong directory: {}".format(len(wrong_dir)))

    return fail_paths, wrong_dir


def validate_parts(original_list, counterpart_ending:str, _raise:bool = False, _print:bool = False) -> list:
    """ Tests if all files have a counterpart. """
    result = []
    for _xml in original_list:
        im_path = str(_xml)
        orignalpart_ending = os.path.basename(im_path).partition(".")[2]
        to_check = im_path.replace("."+orignalpart_ending, counterpart_ending)
        if not os.path.isfile(to_check):
            message = "File at\n {} \nhas no counterpart.".format(im_path)
            result.append(im_path)
            if _raise:
                raise utils.ValidationError(message)
            if _print:
                __print__(message)

    return result


def image_count(_print:bool = True) -> tuple:
    """ Returns the number of xml and jpg files (jpgs, xmls) """
    path = os.path.join(config.MAIN_DATASET_PATH, "**")
    dirs = glob(path, recursive=True)
    jpgs = [el for el in dirs if el.endswith(".jpg")]
    xmls = [el for el in dirs if el.endswith(".xml")]
    if _print:
        __print__("JPG count: {}".format(str(len(jpgs))))
        __print__("XML count: {}".format(str(len(xmls))))
    return len(jpgs), len(xmls)

def plot_imbalance():
    """ Plots the class imbalance. """
    samples_count = [len(glob(os.path.join(config.MAIN_DATASET_PATH, el, '*.jpg'))) for el in sorted(os.listdir(config.MAIN_DATASET_PATH))]
    plt.bar(np.arange(len(samples_count)), sorted(samples_count))
    plt.show()

def create_ignore_list():
    """ Creates a list with IDs to be ignored. """
    fail_paths, _ = validate(_raise=False)
    fail_paths = [el[el.rindex(os.path.sep)+1:el.rindex('.')] for el in fail_paths]
    with open("ignore.txt", "w") as file:
        file.write('\n'.join(fail_paths))


def classes_count(_set:str = 'clean', _plot:bool = False):
    """ Prints further information for the classes. """
    path = os.path.join(config.MAIN_DATASET_PATH,"**") if _set=='clean' else os.path.join(config.NOISY_DATA_SETS[0],"**")
    descriptions = glob(os.path.join(path, "*.xml"),recursive=True)
    class_count_dict = defaultdict(int)
    class_name_dict = defaultdict(str)
    failed_list =  []
    for description in descriptions:
        filename = os.path.basename(description)
        if filename[:filename.rindex('.')] in config.IGNORE_LIST:
            continue
        try:
            doc = xml.dom.minidom.parse(description)
            classid = doc.getElementsByTagName('ClassId')[0].childNodes[0].data
            class_count_dict[classid] += 1
            class_name_dict[classid] = doc.getElementsByTagName('Species')[0].childNodes[0].data
        except xml.parsers.expat.ExpatError:
            failed_list.append(description)
            continue
    print("Failed to parse: {}".format(len(failed_list)))

    print("Most Data:")
    class_counts = class_count_dict.values()
    for key, value in sorted(list(class_count_dict.items()),key=lambda v: v[1], reverse=True)[:5]:
        print("Name: {}".format(class_name_dict[key]))
        print("Count: {}".format(value))
    print("Image count: {}".format(sum(class_counts)))
    print("Average count: {}".format(round(np.average(list(class_counts)))))
    print("Min count: {}".format(min(class_counts)))
    print("Mode count: {}".format(scipy.stats.mode(list(class_counts)).mode[0]))
    print("Class count: {}".format(len(class_name_dict.keys())))
    print("Classes with one sample: {}".format(len([el for el in class_count_dict.values() if int(el) == 1])))
    if _plot:
        class_count_dict = OrderedDict(sorted(class_count_dict.items(), key=lambda v: v[1], reverse=True))
        plt.bar(range(len(class_count_dict)), list(class_count_dict.values()), align='center')
        plt.xticks(range(len(class_count_dict)), list(class_count_dict.keys()))
        plt.show()


def single_sample_classes():
    """ Prints the classes with only one train sample. """
    train_dirs = glob(os.path.join(config.MAIN_DATASET_PATH,'*'))
    single_classes = []
    for _dir in train_dirs:
        if len(glob(os.path.join(_dir, '*.jpg'))) == 1:
            single_classes.append(_dir)
    print(single_classes)


def test_infos():
    """ Prints information ont the test set. """
    jpg_files = glob(os.path.join(config.TESTSET_PATH, '*.jpg'))
    xml_files = glob(os.path.join(config.TESTSET_PATH, '*.xml'))
    print("{} test images".format(len(jpg_files)))
    dic = defaultdict()

    for desc in xml_files:
        obsId = ET.parse(desc).find('ObservationId').text
        try:
            dic[obsId] += 1
        except KeyError:
            dic[obsId] = 1

    print("There are {} unique Observations".format(len(dic.keys())))
    print("Avg. imgs per Obs: {}".format(round(np.average(list(dic.values())), 2)))
    print("Max. imgs per Obs: {}".format(max(dic.values())))


def show_ims_with(*oids:str):
    """ Shows the alle images for the given observation ids. """
    xmls = glob(os.path.join(config.TESTSET_PATH, '*.xml'))
    jpgs = [(ET.parse(el).find('ObservationId').text, el.replace('.xml', '.jpg')) for el in xmls]
    for oid in oids:
        els = utils.find(lambda el: el[0] == oid, jpgs)
        print("Images for {}:".format(oid))
        els = [el[1] for el in els]
        print("\n".join(els))


def handle_args(**kwargs):
    """ Handles the command line arguments. """
    if kwargs['class_count']:
        classes_count(_set=kwargs['class_count'])
    if kwargs['single_sample_classes']:
        single_sample_classes()
    if kwargs['invalid_count']:
        invalid_count_all()
    if kwargs['test_infos']:
        test_infos()
    if kwargs['obs_ims']:
        show_ims_with(*kwargs['obs_ims'])


def main():
    """ Entry point """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--class_count',
        choices=['clean','noisy'],
        help=classes_count.__doc__)
    parser.add_argument(
        '-s',
        '--single_sample_classes',
        action='store_true',
        help=single_sample_classes.__doc__)
    parser.add_argument(
        '-i',
        '--invalid_count',
        action='store_true',
        help=invalid_count_all.__doc__)
    parser.add_argument(
        '-t',
        '--test_infos',
        action='store_true',
        help=test_infos.__doc__)
    parser.add_argument(
        '-o',
        '--obs_ims',
        nargs='+',
        type=str,
        help=show_ims_with.__doc__)

    args = parser.parse_args()
    handle_args(**vars(args))


if __name__ == '__main__':
    main()
