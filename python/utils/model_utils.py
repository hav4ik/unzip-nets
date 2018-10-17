import os
import numpy as numpy
import tensorflow as tf

import custom.models


def load_model(model, weights=None, params={}):
    """loads a model either from module or from a checkpoint
    """
    if hasattr(custom.models, model):
        model_definition = getattr(custom.models, model)
        loaded_model = model_definition(**params)
    else:
        pass

    return loaded_model


def get_variables(name_scope):
    """Gets all variables in the namescope
    """
    variables = []
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope=name_scope):
        variables.append(var)
    return variables

