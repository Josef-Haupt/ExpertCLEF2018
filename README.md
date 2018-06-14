# ExpertCLEF2018

This is the third place of the [ExpertCLEF2018](http://www.imageclef.org/node/231) competition.

## Installation

Install the listed packages and their dependencies for Python 3.6, as well as [CUDA](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html) + [CuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html).

* pandas 0.19.2
* Pillow 5.1.0
* Keras 2.1.2
* Numpy 1.11.3
* OpenCV 3.1.0

Download the training and test data.
Afterwards create a config.xml file in the src directory and fill it with the paths of the datasets, like shown here:

```xml
config.xml:
<?xml version="1.0" encoding="UTF-8"?>
<config>
    <datasets>
        <clean>
            <dataset path="/datasets/ExpertCLEF2018/PlantCLEF2017Train1EOL/data" />
        </clean>
        <noisy>
            <dataset path="/datasets/ExpertCLEF2018/PlantCLEF2017Train2Web/web/data" />
        </noisy>
        <all path="/datasets/ExpertCLEF2017/ExpertCLEF2018EOL_Web/data" />
    </datasets>
    <validation path="/datasets/ExpertCLEF2018/PlantCLEF2017Train1EOL/valid" />
    <testset path="/datasets/ExpertCLEF2018/ExpertCLEF2018Test/test" />
    <models path="/datasets/ExpertCLEF2018/models" />
</config>
```

or use the following command to generate such a file with the given paths already inserted:

    python config.py -c <train_path> <val_path> <models_path> <test_path>

**Note:** The `<all />` tag is not necessary and will not be generated automatically.
The trained models and submissions will be saved to the models path.
Split a validation set from the clean dataset. This will create a *valid* directory inside the clean dataset directory.

    python config.py -s <percentage_per_class>

Some images inside the noisy data set will lead to an Exception while training use the following command to sort out invalid images:

    python config.py -r <absolute_path>

**Note:** This can take a really long time, since the noisy set holds 1M+ images.

## Training

To train a model type:

    python train.py <modelname> ...

Our best result was achieved with an ensamble of three CNNs, nameley a ResNet50 and two DenseNet201.
The DenseNet, which was trained on the noisy dataset, used with the following configuration:

    python train.py densenet201 -b 16 -s 10000 -e 1000 -n -a -m 0.9 -l 0.001 --train_mode single_noisy

**Note:** This configartion needs approx. 8 GB of GPU memory.

To use another model as pretraining, use the following command, with the a path that is relativ to the in the xml file specified models path, the directory should contain a *weights.h5*.

    python train.py densenet201 ... --pretrained densenet201/run1

We generate new weights using the error rates of the densenets on the different classes of the training set:

    python predict.py densenet201/run1 --predict_test /datasets/ExpertCLEF2017/PlantCLEF2017EOL/data --generate_weights

Further DenseNets were trained which used the newly generated weights:

    python train.py densenet201 -b 16 -s 10000 -e 200 -n -a -m 0.9 -l 0.001 \
    --train_mode single_noisy --pretrained densenet201/run1 --weight_mode densenet201/run1

This procedure was repeated for a maximum of three times.

## Additional Notes

To use the test mode you will have to create a testing.xml file, you can copy the config.xml file and edit the paths, here is an example:

```xml
testing.xml:
<?xml version="1.0" encoding="UTF-8"?>
<config>
    <datasets>
        <clean>
            <dataset path="/datasets/ExpertCLEF2018/PlantCLEF20171TrainEOL/testing/data" name="PlantCLEF2017Train1EOL"/>
        </clean>
        <noisy>
        </noisy>
    </datasets>
    <validation path="/datasets/ExpertCLEF2018/PlantCLEF20171TrainEOL/testing/valid" />
    <testset path="" />
    <models path="/datasets/ExpertCLEF2018/models/test" />
</config>
```

Now to Create a smaller train and validation set for testing type:

    python config.py -t <Classes> <Val-samples> <min-class-samples>

to move the validationfiles back into the trainingset type:

    python config.py -u
