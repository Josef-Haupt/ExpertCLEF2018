""" Trainiert das übergebene Modell anhand der Konfigurationsdatei. """
import argparse
import os
import warnings
from contextlib import suppress
from glob import glob
from inspect import getmembers, isfunction

import keras
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator as IDG

import config
import models
import utils

SEED = 7

if config.PC_NAME == 'rick':
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    conf = tf.ConfigProto()
    conf.gpu_options.per_process_gpu_memory_fraction = 0.35 # o.3 reicht für 64 BS inception und 0.28 für 16 BS Dense
    set_session(tf.Session(config=conf))


def top3_acc(y_true, y_pred):
    """ Top-3 """
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)


@utils.timeit_formated(utils.format_time)
def train(model: keras.models.Model,
          optimizer:dict,
          save_path:str,
          train_dir:str,
          valid_dir:str,
          batch_size:int = 32,
          epochs:int = 10,
          samples_per_epoch=1000,
          pretrained=None,
          augment:bool = True,
          weight_mode=None,
          verbose=0,
          **kwargs):
    """ Trains the model with the given configurations. """
    shape = model.input_shape[1:3]
    optimizer_cpy = optimizer.copy()
    shared_gen_args = {
        'rescale': 1./255, # to preserve the rgb palette
    }
    train_gen_args = {}
    if augment:
        train_gen_args = {
            "fill_mode": 'reflect',
            'horizontal_flip': True,
            'vertical_flip': True,
            'width_shift_range': .15,
            'height_shift_range': .15,
            'shear_range': .5,
            'rotation_range': 45,
            'zoom_range': .2,
        }
    gen = IDG(**{**shared_gen_args, **train_gen_args})
    gen = gen.flow_from_directory(train_dir, target_size=shape, batch_size=batch_size, seed=SEED)

    val_count = len(glob(os.path.join(valid_dir, '**', '*.jpg'), recursive=True))
    valid_gen = IDG(**shared_gen_args)

    optim = getattr(keras.optimizers, optimizer['name'])
    if optimizer.pop('name') != 'sgd':
        optimizer.pop('nesterov')
    schedule = optimizer.pop('schedule')
    if schedule == 'decay' and 'lr' in optimizer.keys():
        initial_lr = optimizer.pop('lr')
    else:
        initial_lr = 0.01
    optim = optim(**optimizer)

    callbacks = [
        utils.checkpoint(save_path),
        utils.csv_logger(save_path),
    ]

    if pretrained is not None:
        if not os.path.exists(pretrained):
            raise FileNotFoundError()

        model.load_weights(pretrained, by_name=False)
        if verbose == 1:
            print("Loaded weights from {}".format(pretrained))

    if optimizer_cpy['name'] == 'sgd':
        if schedule == 'decay':
            callbacks.append(utils.step_decay(epochs, initial_lr=initial_lr))
        elif schedule == 'big_drop':
            callbacks.append(utils.constant_schedule())

    model.compile(optim, loss='categorical_crossentropy', metrics=['accuracy', top3_acc])

    create_xml_description(
        save=os.path.join(save_path, 'model_config.xml'),
        title=model.name,
        epochs=epochs,
        batch_size=batch_size,
        samples_per_epoch=samples_per_epoch,
        augmentations=augment,
        schedule=schedule,
        optimizer=optimizer_cpy,
        **kwargs)

    if weight_mode:
        class_weights = [ [key, value] for key, value in weight_mode.items() ]
        filen = os.path.join(save_path, 'class_weights.npy')
        np.save(filen, class_weights)

    h = None # has to be initialized here, so we can reference it later
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h = model.fit_generator(
                gen,
                steps_per_epoch=samples_per_epoch/batch_size,
                epochs=epochs,
                validation_data=valid_gen.flow_from_directory(valid_dir, target_size=shape, batch_size=batch_size, seed=SEED),
                validation_steps=val_count/batch_size,
                callbacks=callbacks,
                class_weight=weight_mode,
                verbose=2)
    except KeyboardInterrupt:
        save_results(
            verbose=1,
            save_path=save_path,
            model=model,
            hist=h)
        return

    save_results(
        verbose=1,
        save_path=save_path,
        model=model,
        hist=h)


def save_results(verbose=0, **kwargs):
    """ Saves all arguments inside a configuration xml. """
    weights_save_path = os.path.join(kwargs['save_path'], 'weights.h5')
    model_save_path = os.path.join(kwargs['save_path'], 'model.h5')
    kwargs['model'].save_weights(weights_save_path)
    kwargs['model'].save(model_save_path)

    if verbose == 1:
        print("Model saved to: {}".format(model_save_path))
        print("Weights saved to: {}".format(weights_save_path))

    with suppress(AttributeError):
        history_save_path = os.path.join(kwargs['save_path'], 'history.csv')
        df = pd.DataFrame(kwargs['hist'].history)
        df.to_csv(history_save_path)
        if verbose == 1:
            print("History saved to: {}".format(history_save_path))


def create_xml_description(save=None, title=None, **kwargs) -> list:
    """ schreibt alle Argumente mitsamt Value in eine xml-datei. """
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<description title="{}">'.format(title if title else ''))

    for key, value in kwargs.items():
        lines.append('<{}>'.format(key))
        if not isinstance(value, dict):
            lines.append('{}'.format(value))
        else:
            for name, info in value.items():
                lines.append('<{}>'.format(name))
                lines.append('{}'.format(info))
                lines.append('</{}>'.format(name))
        lines.append('</{}>'.format(key))
    lines.append('</description>')

    if save:
        with open(save, 'w') as description:
            description.write("\n".join(lines))
        print("Saved description to: {}".format(save))

    return lines


def modelname(string:str) -> str:
    """ Tests if the model is already implemented in models.py """
    possible_models = [ el[0] for el in getmembers(models, predicate=isfunction) ]
    string = string.lower()
    if string not in possible_models:
        raise argparse.ArgumentError('modelname', 'The model has not been implemented.')
    return string


def handle_args(**kwargs):
    """ Handles the commandline arguments. """
    shape = (256,256,3)
    if kwargs['train_mode'] == 'single_clean':
        train_dir = config.MAIN_DATASET_PATH if not kwargs['test_mode'] else config.TEST_DATASET_PATH
    elif kwargs['train_mode'] == 'single_noisy':
        train_dir = config.NOISY_DATA_SETS[0]
    elif kwargs['train_mode'] == 'all':
        train_dir = config.COMBINED_TRAIN_SET
    classes = len([el for el in glob(os.path.join(train_dir, '*')) if os.path.isdir(el)])
    mp = config.MODELS_PATH if not kwargs['test_mode'] else config.TEST_MODELS_PATH

    if not kwargs['continue']:
        model = getattr(models, kwargs['modelname'])
        model = model(shape, classes)
    else:
        from keras.models import load_model
        full_path = os.path.join(mp, kwargs['modelname'].lower(), kwargs['continue'], 'model.h5')
        model = load_model(full_path, custom_objects={'top3_acc': top3_acc})

    kwargs.pop('modelname')
    kwargs.pop('continue')

    optimizer = {
        'name': kwargs.pop('optimizer'),
        'decay': kwargs.pop('decay', None),
        'nesterov': kwargs.pop('nesterov', None),
        'momentum': kwargs.pop('momentum', None),
        'lr': kwargs.pop('lr', None),
        'schedule': kwargs.pop('sched', None),
    }
    optimizer = {key: value for key, value in optimizer.items() if value is not None}
    kwargs.update({'optimizer': optimizer})

    if kwargs['pretrained']:
        kwargs['pretrained'] = utils.path_to_abs(kwargs['pretrained'], mp)
        kwargs['pretrained'] = os.path.join(kwargs['pretrained'], 'weights.h5')

    if kwargs['weight_mode'] == 'samples':
        samples_count = [len(glob(os.path.join(train_dir, el, '*.jpg'))) for el in sorted(os.listdir(train_dir))]
        avg = np.average(samples_count)
        weights = [float(avg/el) for el in samples_count]
        weights = dict(enumerate(weights))
        kwargs['weight_mode'] = weights
    elif kwargs['weight_mode'] == 'none':
        kwargs['weight_mode'] = None
    else:
        filen = os.path.join(kwargs['weight_mode'], 'generated_weights.npy')
        path = utils.path_to_abs(filen, mp)
        if not os.path.exists(path):
            raise FileNotFoundError('There were no weights inside the model directoy.')
        weights = dict(np.load(path))
        kwargs['weight_mode'] = weights

    try:
        save_path = os.path.dirname(os.path.dirname(full_path))
    except UnboundLocalError:
        save_path = os.path.join(mp, model.name)
        os.makedirs(save_path, exist_ok=True)

    save_path = utils.complete_path(save_path, 'run')
    os.makedirs(save_path, exist_ok=True)

    valid_dir = config.VALIDATION_PATH if not kwargs['test_mode'] else config.TEST_VALIDATION_PATH

    kwargs.pop('test_mode')
    train(model, save_path=save_path, valid_dir=valid_dir, train_dir=train_dir, **kwargs)


def main():
    """ Entry point """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'modelname',
        type=modelname,
        metavar='MODEL_NAME',
        help="Name of the model which will be used for training.")
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=32,
        help="Spezifies the batch size.")
    parser.add_argument(
        '-s',
        '--samples_per_epoch',
        type=int,
        default=1000,
        help="Spezifies the number of samples per epoch.")
    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        default=10,
        help="The number of training epochs.")
    parser.add_argument(
        '-o',
        '--optimizer',
        type=str,
        default='sgd',
        choices=['sgd','adam','adadelta','adagrad','rmsprop'],
        help="The optimization algorithm to be used while training.")
    parser.add_argument(
        '-n',
        '--nesterov',
        action='store_true',
        help="Will be ignored if optimizer is anything else than 'sgd'.")
    parser.add_argument(
        '-d',
        '--decay',
        type=float,
        help="Spezifies the decay for the optimizer.")
    parser.add_argument(
        '-l',
        '--lr',
        type=float,
        help="Spezifies the lear rate to be used.")
    parser.add_argument(
        '-m',
        '--momentum',
        type=float,
        help="Will be ignored if the optimizer does not support momentum.")
    parser.add_argument(
        '-t',
        '--test_mode',
        action='store_true',
        help="Activats the test mode, only paths from testing.xml will be used.")
    parser.add_argument(
        '-a',
        '--augment',
        action='store_true',
        help="Aktivates data augmentation.")
    parser.add_argument(
        '--train_mode',
        choices=['single_clean','single_noisy','all'],
        default='single_clean',
        help="Defines which training set should be used.")
    parser.add_argument(
        '--sched',
        choices=['none','decay','big_drop'],
        default='none',
        help='Will be ignored if optimizer is not sgd.')
    parser.add_argument(
        '-p',
        '--pretrained',
        type=str,
        help='Has to be relativ to the models directory.')
    parser.add_argument(
        '-w',
        '--weight_mode',
        type=str,
        default='none',
        help="""'none' will not add any spezial weight configuration,
        'samples' will balance the class weights base on the sample count per class,
        You can also use the generated weights inside a models directory, it has to be relativ to model path.""")
    parser.add_argument(
        '-c',
        '--continue',
        type=str,
        help='Has to be relativ to model directory minus the model part (densenet, etc).')
    parser.add_argument('-v', '--verbose', type=int, default=0)

    args = parser.parse_args()
    handle_args(**vars(args))


if __name__ == '__main__':
    main()
