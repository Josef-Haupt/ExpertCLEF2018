""" Module is used for prediction and evaluation. """
import argparse
import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from glob import glob

import keras
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

import config
import train
import utils


def predict(model: keras.models.Model, test_path:str):
    """ Predicts the files inside *test_path* and saves the prediction to *pred_path*. """
    predictions = _predict_core(model, test_path)
    return predictions


def _predict_core(model, path):
    """ Predicts all files inside *path* with the given *model*. """
    gen = ImageDataGenerator(rescale=1./255)
    gen = gen.flow_from_directory(
        path,
        target_size=(256, 256),
        class_mode=None,
        shuffle=False)
    predictions = model.predict_generator(gen, verbose=1)
    return predictions


def predict_test(model:keras.models.Model, test_dir, err_path) -> list:
    """ Saves the error rate for every class inside *err_path*, the given *model* gets for the files inside *test_dir*. """
    predictions = _predict_core(model, test_dir)
    err_rates = []
    sample_counts = [len(glob(os.path.join(test_dir, el, "*.jpg"))) for el in sorted(os.listdir(test_dir))]

    if len(predictions) != sum(sample_counts):
        raise ValueError()

    for i, el in enumerate(sample_counts):
        pre = 0 if i == 0 else sum(sample_counts[:i])
        class_samples_preds = predictions[pre:pre+el]
        right = len([el for el in class_samples_preds if i == np.argmax(el)])
        err_rates.append((right*100)/len(class_samples_preds))

    filen = os.path.join(err_path, 'error_rates.npy')
    np.save(filen, err_rates)
    return err_rates


def generate_weights(err_rates, class_weights, save_path:str):
    """
    Generates a new weight configuration based on the *err_rates* and the current *class_weights*.
    Saves the configuration to *save_path*.
    """
    class_weights = sorted(class_weights, key=lambda el: el[0])
    new_weights = []
    vals = []

    for i, el in enumerate(class_weights):
        key, value = el
        new_val = value * (1.85 - err_rates[i]/100)
        vals.append(new_val)
        new_weights.append([key, new_val])

    np.save(os.path.join(save_path, 'generated_weights.npy'), new_weights)


@utils.timeit
def make_submission(preds, save_path, percentage=[], top=10):
    """ Creates a valid submission based on the given *preds*. """
    observationIds = [ET.parse(os.path.join(config.TESTSET_PATH, file)).find('ObservationId').text
                      for file in sorted(os.listdir(config.TESTSET_PATH)) if file.endswith('.xml')]
    classIds = [el for el in sorted(os.listdir(config.MAIN_DATASET_PATH))]
    obs_dic = defaultdict(np.array)
    percentage = utils.softmax2(percentage) if percentage else utils.softmax2([1 for _ in range(len(preds))])
    obs_count = Counter(observationIds)

    for i, model_pred in enumerate(preds):
        factor = percentage[i]
        for j, pred in enumerate(model_pred):
            obs_id = observationIds[j]
            try:
                obs_dic[obs_id] += pred * factor * (1/obs_count[obs_id])
            except TypeError:
                obs_dic[obs_id] = pred * factor * (1/obs_count[obs_id])

    obs_dic = {key:list(zip(classIds, val)) for key, val in obs_dic.items()}
    obs_dic = {key:sorted(val, key=lambda el: el[1], reverse=True)[:top] for key, val in obs_dic.items()}

    lines = []
    # lines.append("ObservationId;ClassId;Probability;Rank") # Examples has no head line
    for o_id, val in obs_dic.items():
        for i, el in enumerate(val):
            c_id, prob = el
            line = "{observationId};{classId};{probability};{rank}".format(observationId=o_id, classId=c_id, probability=prob, rank=(i+1))
            lines.append(line)

    file_path = os.path.join(save_path, "submission.csv")
    with open(file_path, mode='w') as file:
        file.write("\n".join(lines))
    np.save(os.path.join(os.path.dirname(file_path), 'percentages.npy'), percentage)


def _predict_test_set(model, save_preds:str=None):
    """ The given *model* predicts the test set and saves them to *save_preds*. """
    preds = _predict_core(model, os.path.dirname(config.TESTSET_PATH))
    if save_preds is not None:
        np.save(os.path.join(save_preds, 'testset_preds.npy'), preds)
    return preds


@utils.timeit_formated(utils.format_time)
def evaluate(model, path, save_path=None):
    """ Evaluates the given *model* with the files inside *path* and saves the result. """
    gen = ImageDataGenerator(rescale=1./255)
    gen = gen.flow_from_directory(
        path,
        target_size=(256,256),
        shuffle=False)
    metr = model.evaluate_generator(gen)
    print("Loss: {metr[0]:.4f}\nAcc: {metr[1]:.2%}\nTop3-Acc: {metr[2]:.2%}".format(metr=metr))
    if save_path:
        pd.DataFrame({'loss': [metr[0]], 'acc': [metr[1]], 'top3_acc': [metr[2]]}).to_csv(os.path.join(save_path, 'eval.csv'))
    return metr


def handle_args(**kwargs):
    """ Handles the command line arguments. """
    predictions = []
    preds = []
    for modelpath in kwargs['modelpath']:
        if os.path.isabs(modelpath):
            model = load_model(modelpath)
        else:
            if kwargs['test_mode']:
                modelpath = os.path.join(config.TEST_MODELS_PATH, modelpath, 'model.h5')
            else:
                modelpath = os.path.join(config.MODELS_PATH, modelpath, 'model.h5')

        if kwargs['legacy_weights']:
            data_set_path = config.NOISY_DATA_SETS[0]
            samples_count = [len(glob(os.path.join(data_set_path, el, '*.jpg'))) for el in sorted(os.listdir(data_set_path))]
            max_count = np.average(samples_count)
            weights = [float(max_count/el) for el in samples_count]
            weights = [[i,el] for i, el in enumerate(weights)]
            np.save(os.path.join(os.path.dirname(modelpath), 'class_weights.npy'), weights)

        if kwargs['std_weights']:
            data_set_path = config.NOISY_DATA_SETS[0]
            classes = os.listdir(config.NOISY_DATA_SETS[0])
            weights = [[i,1.] for i in range(len(classes))]
            np.save(os.path.join(os.path.dirname(modelpath), 'class_weights.npy'), weights)

        fp = os.path.join(os.path.dirname(modelpath), 'testset_preds.npy')

        if kwargs['predict'] is not None or kwargs['predict_test'] or (kwargs['make_submission'] is not None and not os.path.exists(fp)) or kwargs['eval']:
            model = load_model(modelpath, custom_objects={'top3_acc': train.top3_acc})

        if kwargs['predict'] is not None:
            path = kwargs['predict'][0]
            preds.append(predict(model, path))

        if kwargs['make_submission'] is not None:
            if not os.path.exists(fp):
                temp = _predict_test_set(model, save_preds=os.path.dirname(modelpath))
            else:
                temp = np.load(fp)

            predictions.append(temp)

        if kwargs['predict_test']:
            predict_test(model, kwargs['predict_test'], os.path.dirname(modelpath))

        if kwargs['generate_weights']:
            err_rates = np.load(os.path.join(os.path.dirname(modelpath), 'error_rates.npy'))
            class_weights = np.load(os.path.join(os.path.dirname(modelpath), 'class_weights.npy'))
            generate_weights(err_rates, class_weights, os.path.dirname(modelpath))

        if kwargs['eval']:
            eval_dir = config.VALIDATION_PATH
            evaluate(model, eval_dir, save_path=os.path.dirname(modelpath))

    if kwargs['predict'] is not None:
        percentages = utils.softmax2(list(map(float,kwargs['predict'][1:])))
        preds = np.sum([pred*percentages[i] for i, pred in enumerate(preds)], axis=0)
        classes = sorted(os.listdir(config.MAIN_DATASET_PATH))

        pred_files = sorted(glob(os.path.join(kwargs['predict'][0], '**', '*.[jj][pp][ge]*')))

        for i, pred_file in enumerate(pred_files):
            print(pred_file)
            elems = sorted(zip(classes, preds[i]), key=lambda el: el[1], reverse=True)[:3]
            for elem in elems:
                classId, prop = elem
                print(classId)
                print(prop)
                print("-")

    if kwargs['make_submission'] is not None:
        dir_name = utils.generate_dirname(kwargs['modelpath'])
        save_path = os.path.join(config.MODELS_PATH, 'submissions', dir_name)
        os.makedirs(save_path, exist_ok=True)
        make_submission(predictions, save_path, percentage=kwargs['make_submission'])

def main():
    """ Entry point """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'modelpath',
        type=str,
        nargs='+',
        metavar='MODELPATH',
        help="Has to be relative to the specified models path in the config-file or an absolute path")
    parser.add_argument(
        '-p',
        '--predict',
        nargs="*",
        help='The first argument has to be a path with exactly 1 directory, with the images to predict inside.')
    parser.add_argument(
        '-t',
        '--test_mode',
        action='store_true')
    parser.add_argument(
        '--make_submission',
        nargs="*",
        type=float,
        metavar="PERCENTAGE",
        help=""""Takes percentages for the weighted vote when combining the test predictions,i
            if none is spezified all preds will count equally.""")
    parser.add_argument(
        '--predict_test',
        type=str,
        help="Predicts the images inside the given directory and generates the error rates.")
    parser.add_argument(
        '-g',
        '--generate_weights',
        action='store_true',
        help="Generates a new weight configuration with the error rates and class weight inside the model directory.")
    parser.add_argument(
        '--legacy_weights',
        action='store_true',
        help="""Generates the legacy weights and saves them inside the model directory,
        the legacy weights weight from the avg sample count per class.""")
    parser.add_argument(
        '--std_weights',
        action="store_true",
        help="Saves the standart weights inside the model directory.")
    parser.add_argument(
        '-e',
        '--eval',
        action='store_true',
        help="Evaluates the models based on the validation set.")

    args = parser.parse_args()
    handle_args(**vars(args))


if __name__ == '__main__':
    main()
