import keras.backend as K
from keras.models import Model
import numpy as np
import time


def _evaluate(model: Model, nodes_to_evaluate, x_mn, x_api, x_token, y=None):
    symb_inputs = (model._feed_inputs)
    print(symb_inputs)
    f = K.function(symb_inputs, nodes_to_evaluate)
    x = [x_mn, x_api, x_token]
    print(x)
    x_, y_, sample_weight_ = model._standardize_user_data(x, y)
    return f(x_ + y_)
    #if not model._is_compiled:
        #if model.name in ['vgg16', 'vgg19', 'inception_v3', 'inception_resnet_v2', 'mobilenet_v2', 'mobilenetv2']:
            #print('Transfer learning detected. Model will be compiled with ("categorical_crossentropy", "adam").')
            #print('If you want to change the default behaviour, then do in python:')
            #print('model.name = ""')
            #print('Then compile your model with whatever loss you want: https://keras.io/models/model/#compile.')
            #print('If you want to get rid of this message, add this line before calling keract:')
            #print('model.compile(loss="categorical_crossentropy", optimizer="adam")')
            #model.compile(loss='categorical_crossentropy', optimizer='adam')
        #else:
            #print('Please compile your model first! https://keras.io/models/model/#compile.')
            #print('If you only care about the activations (outputs of the layers), '
                  #'then just compile your model like that:')
            #print('model.compile(loss="mse", optimizer="adam")')
            #raise Exception('Compilation of the model required.')

    #def eval_fn(k_inputs):
        #return K.function(k_inputs, nodes_to_evaluate)(model._standardize_user_data(x, y))


def get_gradients_of_trainable_weights(model, x, y):
    nodes = model.trainable_weights
    nodes_names = [w.name for w in nodes]
    return _get_gradients(model, x, y, nodes, nodes_names)


def get_gradients_of_activations(model, x, y, layer_name=None):
    nodes = [layer.output for layer in model.layers if layer.name == layer_name or layer_name is None]
    nodes_names = [n.name for n in nodes]
    return _get_gradients(model, x, y, nodes, nodes_names)


def get_gradients_of_activations_input(model, x, y, layer_name=None):
    nodes = [layer.input for layer in model.layers if layer.name == layer_name or layer_name is None]
    nodes_names = [n.name for n in nodes]
    return _get_gradients(model, x, y, nodes, nodes_names)


def _get_gradients(model, x, y, nodes, nodes_names):
    if model.optimizer is None:
        raise Exception('Please compile the model first. The loss function is required to compute the gradients.')
    grads = model.optimizer.get_gradients(model.total_loss, nodes)
    gradients_values = _evaluate(model, grads, x, y)
    result = dict(zip(nodes_names, gradients_values))
    return result


def get_activations(model, x, layer_name=None):
    nodes = [layer.output for layer in model.layers if layer.name == layer_name or layer_name is None]
    # we process the placeholders later (Inputs node in Keras). Because there's a bug in Tensorflow.
    input_layer_outputs, layer_outputs = [], []
    [input_layer_outputs.append(node) if 'input_' in node.name else layer_outputs.append(node) for node in nodes]
    activations = _evaluate(model, layer_outputs, x, y=None)
    activations_dict = dict(zip([output.name for output in layer_outputs], activations))
    activations_inputs_dict = dict(zip([output.name for output in input_layer_outputs], x))
    result = activations_inputs_dict.copy()
    result.update(activations_dict)
    return result


def get_activations_single_layer(model, x_mn, x_api, x_token, layer_name=None):
    nodes = [layer.output for layer in model.layers if layer.name == layer_name or layer_name is None]
    # we process the placeholders later (Inputs node in Keras). Because there's a bug in Tensorflow.
    input_layer_outputs, layer_outputs = [], []
    [input_layer_outputs.append(node) if 'input_' in node.name else layer_outputs.append(node) for node in nodes]
    activations = _evaluate(model, layer_outputs, x_mn, x_api, x_token, y=None)
    activations_dict = dict(zip([output.name for output in layer_outputs], activations))
    activations_inputs_dict = dict(zip([output.name for output in input_layer_outputs], x_mn, x_api, x_token))
    result = activations_inputs_dict.copy()
    result.update(activations_dict)
    return np.squeeze(list(result.values())[0])


def display_activations(activations):
    import matplotlib.pyplot as plt
    max_rows = 8
    max_columns = 8
    for layer_name, first in activations.items():
        print(layer_name, first.shape, end=' ')
        if first.shape[0] != 1:
            print('-> Skipped. First dimension is not 1.')
            continue
        if len(first.shape) <= 2:
            print('-> Skipped. 2D Activations.')
            continue
        print('')
        fig = plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.title(layer_name)
        for i in range(1, min(max_columns * max_rows + 1, first.shape[-1] + 1)):
            img = first[0, :, :, i - 1]
            fig.add_subplot(max_rows, max_columns, i)
            plt.imshow(img)
            plt.axis('off')
        plt.show()

def get_gradients_function(model, layer_name=None):
    nodes = [layer.input for layer in model.layers if layer.name == layer_name or layer_name is None]
    nodes_names = [n.name for n in nodes]
    if model.optimizer is None:
        raise Exception('Please compile the model first. The loss function is required to compute the gradients.')
    grads = model.optimizer.get_gradients(model.total_loss, nodes)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    return f, nodes_names

def cal_gradient(model,f,x,y,nodes_names):
    x_, y_, sample_weight_ = model._standardize_user_data(x, y)
    gradients_values = f(x_ + y_ + sample_weight_)
    result = dict(zip(nodes_names, gradients_values))
    return result





