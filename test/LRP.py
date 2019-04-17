import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from sklearn.metrics import classification_report
from keras import backend as K
import matplotlib.pyplot as plt  
import numpy as np
import pickle
import utils



def rot180(ndarray):
    rot= np.flip(np.flip(ndarray, 1), 0)
    return rot

class LayerwiseRelevancePropagation:
    def __init__(self, model_name='32CNN.h5', alpha=2, epsilon=1e-9):
        self.model = load_model(model_name)
        self.alpha = alpha
        self.beta = 1 - alpha
        self.epsilon = epsilon

        self.names, self.activations, self.weights = utils.get_model_params(self.model)
        self.num_layers = len(self.names)

        self.relevance= self.compute_relevances()
        self.lrp_runner = K.function(inputs=[self.model.input, ], outputs=[self.relevance, ])


    def print_model(self):
        print('---------------------------------')
        print(self.names[0])

        print(self.model.input) #Tensor("conv1d_93_input:0", shape=(?, 120, 1), dtype=float32)
        #print(self.model.output) #("dense_31/Softmax:0", shape=(?, 16), dtype=float32)
        print(self.activations[0]) #Tensor("conv1d_93_input:0", shape=(?, 120, 1), dtype=float32)
        print(self.relevance) #Tensor("mul_11:0", shape=(?, 120, 1), dtype=float32)
        print(self.weights[0])
        print('---------------------------------')
        print('==============names==============')      
        for i in self.names:
            print('##', len(self.names))
            print (i)
        print('==============activation==============')  
        for x in self.activations:
            print('##', len(self.activations))
            print(type(x))
        print('==============weight==============') 
        for x in self.weights:
            print('##', len(self.weights))
            print(type(x))

    def compute_relevances(self):
        r= self.model.output
        for i in range(self.num_layers-2, -1, -1):
            #print(i+1)
            #print('vist:',self.names[i + 1])
        
            if 'fc' in self.names[i + 1]:
                #print('===================== compute_relevances fc=====================')
                r = self.backprop_fc(self.weights[i + 1][0], self.weights[i + 1][1], self.activations[i], r)
                #print('########fc Relevances: ######## ', r)
            elif 'flatten' in self.names[i + 1]:
                #print('=====================compute_relevances flatten=====================')
                r = self.backprop_flatten(self.activations[i], r)
                #print('########flatten Relevances: ######## ', r)
            elif 'conv' in self.names[i + 1]:
                #print('=====================compute_relevances conv=====================')
                r = self.backprop_conv(self.weights[i + 1][0], self.weights[i + 1][1], self.activations[i], r)
                #print('########conv Relevances: ######## ', r)

        return r


    def first_fc(self, weights, bias, activations, relevances):
        lowest= -1.0
        highest= 1.0    
        V= K.maximum(weights, 0.)
        U= K.minimum(weights, 0.)
        L= activations*0+lowest
        H= activations*0+highest

        Z= K.dot(activations, weights)- K.dot(L, V)-K.dot(H, U)+self.epsilon
        S= relevance/Z
        return(activations*K.dot(S, transpose(weights))- L*K.dot(S, transpose(V)) - H*K.dot(S, transpose(U)))

    def backprop_fc(self, weights, bias, activations, relevances):
        '''
        V= K.maximum(weights, 0.)
        Z = K.dot(activations,V)+ self.epsilon
        S = relevances/Z
        C =  K.dot(S, K.transpose(V))
        print('===================== backprop_fc finished ==========================')
        return (activations*C)
        '''
        w_p = K.maximum(weights, 0.)
        b_p = K.maximum(bias, 0.)

        z_p = K.dot(activations, w_p) + b_p + self.epsilon
        
        s_p = relevances / z_p

        c_p = K.dot(s_p, K.transpose(w_p))

        w_n = K.minimum(weights, 0.)
        b_n = K.maximum(bias, 0.)
        z_n = K.dot(activations, w_n) + b_n - self.epsilon
        s_n = relevances / z_n
        c_n = K.dot(s_n, K.transpose(w_n))
        print('===================== backprop_fc finished ==========================')
        return activations * (self.alpha * c_p + self.beta * c_n)


    def backprop_flatten(self, a, r):
        shape = a.get_shape().as_list()
        shape[0] = -1
        #print('===================== backprop_flatten finished ==========================')
        return K.reshape(r, shape)

    def first_conv(self, w, b, a, r):

        w_p = K.maximum(w, 0.)
        b_p = K.maximum(b, 0.)

        z_p = K.conv1d(a, kernel=w_p, strides=1, padding='valid') + b_p + self.epsilon  

        s_p = r / z_p

        c_p = K.tf.contrib.nn.conv1d_transpose(
                value= s_p,
                filter= w_p, 
                output_shape= K.shape(a),
                stride= 1,
                padding='SAME',
                name=None
                )

        w_n = K.minimum(w, 0.)
        b_n = K.minimum(b, 0.)
        z_n = K.conv1d(a, kernel=w_n, strides=1, padding='valid') + b_n - self.epsilon
        s_n = r / z_n
        c_n = K.tf.contrib.nn.conv1d_transpose(
                value= s_n,
                filter= w_n,
                output_shape= K.shape(a),
                stride= 1,
                padding='SAME',
                name=None
                )

        return a * (self.alpha * c_p + self.beta * c_n)


    '''
        lowest= -1.0
        highest= 1.0
        w_p = K.maximum(w, 0.)
        w_n = K.minimum(w, 0.)
        iself = keras.models.clone_model(self.model)
        nself = keras.models.clone_model(self.model)
        pself = keras.models.clone_model(self.model)

        K.set_value(iself.layers[0].weights[1], np.zeros((16,)))

        K.set_value(nself.layers[0].weights[1], np.zeros((16,)))
        #K.set_value(nself.layers[0].weights[0], w_n)

        K.set_value(pself.layers[0].weights[1], np.zeros((16,)))
        #K.set_value(pself.layers[0].weights[0], w_p)

        X, L, H= a, X*0*lowest, X*0+highest

        zn= K.conv1d(X, kernel=w_n, strides=1, padding='valid')
        zp= K.conv1d(X, kernel=w_p, strides=1, padding='valid')
        z=  K.conv1d(X, kernel=w, strides=1, padding='valid')   



        Z=  K.conv1d(X, kernel=w, strides=1, padding='valid')  
        Z = K.conv1d(a, kernel=w_p, strides=1, padding='valid') + b_p + self.epsilon
        self.lrp_runner = K.function(inputs=[self.model.input, ], outputs=[self.relevance, ])
        
        w_p = K.maximum(w, 0.)
        print('************', w_p)
        b_p = K.maximum(b, 0.)

        z = K.conv1d(a, kernel=w_p, strides=1, padding='valid') + self.epsilon
        zn

        s_p = r / z

        c_p = K.tf.contrib.nn.conv1d_transpose(
                value= s_p,
                filter= w_p, 
                output_shape= K.shape(a),
                stride= 1,
                padding='SAME',
                name=None
                )

        return(a*c_p)
    '''

    def backprop_conv(self, w, b, a, r):

        w_p = K.maximum(w, 0.)
        b_p = K.maximum(b, 0.)
        z_p = K.conv1d(a, kernel=w_p, strides=1, padding='valid') + b_p + self.epsilon  
        s_p = r / z_p
        c_p = K.tf.contrib.nn.conv1d_transpose(
                value= s_p,
                filter= w_p, 
                output_shape= K.shape(a),
                stride= 1,
                padding='SAME',
                name=None
                )
        w_n = K.minimum(w, 0.)
        b_n = K.minimum(b, 0.)
        z_n = K.conv1d(a, kernel=w_n, strides=1, padding='valid') + b_n - self.epsilon
        s_n = r / z_n
        c_n = K.tf.contrib.nn.conv1d_transpose(
                value= s_n,
                filter= w_n,
                output_shape= K.shape(a),
                stride= 1,
                padding='SAME',
                name=None
                )

        return a * (self.alpha * c_p + self.beta * c_n)



    def predict_labels(self, samples):
        return utils.predict_labels(self.model, samples)

    def run_lrp(self, samples):
        lrp_runner = K.function(inputs=[self.model.input, ], outputs=[self.relevance, ])        
        return self.lrp_runner([samples, ])[0]

    def compute_heatmap(self, samples, **kwargs):
        lrp=self.run_lrp(samples)

if __name__ == '__main__':

    Test_X = np.array(pickle.load(open("Test_X.pickle","rb")))
    Test_y = np.array(pickle.load(open("Test_y.pickle","rb")))
    Test_X = Test_X/100.0
    Test_X=np.reshape(Test_X, (Test_X.shape[0], Test_X.shape[1],1))

    lrp=LayerwiseRelevancePropagation()
    #lrp.print_model()

    model=lrp.model
    #sample= np.expand_dims(Test_X[1000], axis=0)
    #print(model.predict_classes(sample))

    ################## test backend function########################################

    get_3rd_layer_output = K.function([model.layers[0].input], [model.layers[3].output])
    layer_output = get_3rd_layer_output([Test_X])[0]

    ################## test backend function########################################