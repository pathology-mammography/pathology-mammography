# Pathological subtype classification of ductal carcinoma with no special type by mammography
This algorthm was based on VGGNet to classify ductal carcinoma with no special type of breast cancer in mammograms.

You can use this algorithm after downloaded hdf5 file from release page.
This algorithm is not for a diagnostic use.

## Install

- Clone this repository.
- Download a hdf5 file (best-performing.hdf5) into the dorectory of "h5files".

hdf5 file can be downloaded from:
https://github.com/pathology-mammography/pathology-mammography/releases


## Usage
Training: `python SGD.py` in the directory of "code". 
Inference: `python prediction.py` in the directory of "code". 

## Enviroment
This algorithm was built in the TensorFlow framework (https://www.tensorflow.org/) with the Keras wrapper library (https://keras.io/).

- tensorflow-gpu 1.10.0
- Keras 2.2.4

## Research paper
This algorithm was published on XXX.
