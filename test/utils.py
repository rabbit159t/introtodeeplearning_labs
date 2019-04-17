import numpy              as np
import matplotlib.pyplot  as plt

def get_model_params(model):
	names, activations, weights = ['input_1'], [model.input, ], [0,]

	for layer in model.layers:
		name = layer.name if layer.name != 'predictions' else 'fc_out'
		names.append(name)
		activations.append(layer.output)
		weights.append(layer.get_weights())

	return names, activations, weights

def predict_labels(model, sample):
	pred = model.predict_classes(sample)
	return pred
