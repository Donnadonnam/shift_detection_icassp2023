SELF-SUPERVISED ENHANCEMENT OF STIMULUS-EVOKED BRAIN RESPONSE DATA
------------------------------------------------------------------
[![Python version 3.6+](https://img.shields.io/badge/python-3.6%2B-brightgreen)](https://www.python.org/downloads/)
[![Tensorflow version 2.3+](https://img.shields.io/badge/Tensorflow-v2.3.0%2B-orange)](https://tensorflow.org)
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](./LICENSE)


This repository contains code and pre-trained model versions of the models
used in the ICASSP 2023 submission "Self-Supervised Enhancement of Stimulus-Evoked Brain Response Data"
by [Bernd Accou](https://gbiomed.kuleuven.be/english/research/50000666/50000672/people/members/00114712), [Hugo Van hamme](https://www.kuleuven.be/wieiswie/en/person/00040707), and [Tom Francart](https://gbiomed.kuleuven.be/english/research/50000666/50000672/people/members/00046624).


# Requirements

The code/models in this repository use [Tensorflow](https://www.tensorflow.org/) version >= 2.3.0 and Python >= 3.6.0.
which can be installed via conda or pip (see also the [Tensorflow installation guide](https://www.tensorflow.org/install)).

Example installation using pip
```bash
pip install tensorflow
```

# Code

Code for all models is stored in [models.py](./models.py).


# Pre-trained models

Pre-trained models can be found in the [pretrained_models](pretrained_models) folder:

*  The [shift detection model](./pretrained_models/shift_detection_model) contains the weights for the full shift detection model, including the enhancement module (multi-view CNN based architecture) and the comparison model.
*  The [subject-independent linear decoder](./pretrained_models/subject_independent_decoder) contains the weights for the subject-independent linear decoder, used in the paper for the downstream speech envelope decoding task.

Both models are saved in tensorflow [SavedModel format](https://www.tensorflow.org/guide/saved_model).

Example code for loading the models:
```python
import tensorflow as tf

# Load the model
shift_detection_model = tf.keras.models.load_model('pretrained_models/shift_detection_model')
# Extract the multi-view CNN based enhancement module
enhancement_module = shift_detection_model.get_layer('multiview_cnn')
# Extract the simple comparison model
simple_comparison_model = shift_detection_model.get_layer('simple_comparison_model')

# Load the subject-independent linear decoder
subject_independent_decoder = tf.keras.models.load_model('pretrained_models/subject_independent_decoder')

```

