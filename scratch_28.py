
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Input, Convolution2D, Activation, MaxPooling2D, \
    Dense, BatchNormalization, Dropout

from keras.optimizers import SGD
import numpy as np

import math as mt
import itertools


num_M=5
num_U=5
N=9000
N_subband=3

MC_power_1=[0.4, 0.6, 0.8,1.0,1.2]
MC_power=[0.4,0.6, 0.8,1.0,1.2]

N_RB=48
N_slot=1
N_subframes=1
sub=[]
slots_slots=[1]
sub.append(mt.floor(N_RB/N_subband))
sub.append(mt.floor(N_RB/N_subband))
sub.append(N_RB-sub[N_subband-2]*(N_subband-1))
#print(sub)
final_comb=[]
power_comb = list(map(list, itertools.product(MC_power_1, repeat=N_subband)))

for i in power_comb:
    temp_sum=0.0
    for j in range(N_subband):
        temp_sum=temp_sum+ i[j]*sub[j]
    if temp_sum <=40 :
        final_comb.append(i)

ln_fl_comb=len(final_comb)
print(final_comb)
aw=final_comb.index([1.2,0.8,0.4])
print("aw----",aw,ln_fl_comb)


data_set = np.load("data_set_5c_final.npz")

g_data_x = data_set['arr_0'].tolist()
g_data_y = data_set['arr_1'].tolist()
g_data_action_ga = data_set['arr_2'].tolist()
g_data_ga_th = data_set['arr_3'].tolist()
g_data_action_wmmse = data_set['arr_4'].tolist()
g_data_wmmse_th = data_set['arr_5'].tolist()
g_data_vector_ga=data_set['arr_6']

print(g_data_action_ga[0])

data_x=[]
data_y=[]
for i in range(len(g_data_vector_ga)):
    temp_x=g_data_vector_ga[i][0:100]
    temp_y = g_data_vector_ga[i][100:115]
    data_x.append(temp_x)
    data_y.append(temp_y)
data_x=np.array(data_x)
data_y=np.array(data_y)
data_y=data_y.astype('int')
final=[]
c=[]
for i in range(len(data_y)):

    for j in range(len(data_y[i])):
#        f=[int(x) for x in list('{0:3b}'.format(a[[i],[j]][0]))]
        #f=[int(x) for x in bin(data_y[[i],[j]][0])[2:].zfill(3)]
        f_=[int(x) for x in bin(data_y[[i],[j]][0])[2:].zfill(3)]
        f= (1-np.array(f_)).tolist()
        f_final=[]
        for k in range(len(f_)):
            f_final.append(f_[k])
            f_final.append(f[k])
        #print(f)
        final.extend(f_final)
    c.append(final)
    final=[]

print(c[0:3])
data_y=np.array(c)
print(data_y)

data_x=data_x.astype('float32')
data_y=data_y.astype('float32')


print(len(data_x[0]),len(data_y[0]))

X_train=data_x[0:round(0.8*len(data_x))]
Y_train=data_y[0:round(0.8*len(data_x))]
X_test=data_x[round(0.8*len(data_x)):len(data_x)]
Y_test=data_y[round(0.8*len(data_x)):len(data_x)]

print(len(Y_test),len(Y_train))
print(Y_test)

# Layer_1
input_img = Input(shape = (100, ))
#distorted_input1 = Dropout(.1)(input_img)
encoded1 = Dense(800, activation = 'sigmoid')(input_img)
encoded1_bn = BatchNormalization()(encoded1)

decoded1 = Dense(100, activation = 'sigmoid')(encoded1_bn)
autoencoder1 = Model(input_img, decoded1)
encoder1 = Model(input_img, encoded1_bn)

# Layer 2
encoded1_input = Input(shape = (800,))
#distorted_input2 = Dropout(.2)(encoded1_input)
encoded2 = Dense(400, activation = 'sigmoid')(encoded1_input)
encoded2_bn = BatchNormalization()(encoded2)
decoded2 = Dense(800, activation = 'sigmoid')(encoded2_bn)

autoencoder2 = Model(encoded1_input, decoded2)
encoder2 = Model(encoded1_input, encoded2_bn)

# Layer 3 - which we won't end up fitting in the interest of time
encoded2_input = Input(shape = (400,))
#distorted_input3 = Dropout(.3)(encoded2_input)
encoded3 = Dense(200, activation = 'sigmoid')(encoded2_input)
encoded3_bn = BatchNormalization()(encoded3)
decoded3 = Dense(400, activation = 'sigmoid')(encoded3_bn)

autoencoder3 = Model(encoded2_input, decoded3)
encoder3 = Model(encoded2_input, encoded3_bn)

# Layer_4
encoded3_input = Input(shape = (200,))
encoded4 = Dense(15*3*2, activation = 'sigmoid')(encoded3_input)

softmax1 = Model(encoded3_input, encoded4)

# Not as Deep Autoencoder
nad_encoded1_da = Dense(800, activation = 'sigmoid')(input_img)
nad_encoded1_da_bn = BatchNormalization()(nad_encoded1_da)
nad_encoded2_da = Dense(400, activation = 'sigmoid')(nad_encoded1_da_bn)
nad_encoded2_da_bn = BatchNormalization()(nad_encoded2_da)
nad_encoded3_da = Dense(200, activation = 'sigmoid')(nad_encoded2_da_bn)
nad_encoded3_da_bn = BatchNormalization()(nad_encoded3_da)


dense1 = Dense(15*3*2, activation='sigmoid')(nad_encoded3_da_bn)

nad_deep_autoencoder = Model(input_img, dense1)

sgd1 = SGD(lr = 5, decay = 0.5, momentum = .85, nesterov = True)
sgd2 = SGD(lr = 5, decay = 0.5, momentum = .85, nesterov = True)
sgd3 = SGD(lr = 5, decay = 0.5, momentum = .85, nesterov = True)

autoencoder1.compile(loss='mse', optimizer = sgd1)
autoencoder2.compile(loss='mse', optimizer = sgd2)
autoencoder3.compile(loss='mse', optimizer = sgd3)

encoder1.compile(loss='mse', optimizer = sgd1)
encoder2.compile(loss='mse', optimizer = sgd1)
encoder3.compile(loss='mse', optimizer = sgd1)
softmax1.compile(loss='mse', optimizer=  sgd1)

nad_deep_autoencoder.compile(loss='mse', optimizer = sgd1)

autoencoder1.fit(X_train, X_train,
                epochs = 8, batch_size = 512,
                validation_split = 0.25,
                shuffle = True)
first_layer_code = encoder1.predict(X_train)
print(first_layer_code.shape)

autoencoder2.fit(first_layer_code, first_layer_code,
                epochs = 8, batch_size = 512,
                validation_split = 0.25,
                shuffle = True)

second_layer_code = encoder2.predict(first_layer_code)
print(second_layer_code.shape)

autoencoder3.fit(second_layer_code, second_layer_code,
                epochs = 8, batch_size = 512,
                validation_split = 0.25,
                shuffle = True)
third_layer_code = encoder3.predict(second_layer_code)
print(third_layer_code.shape)

softmax1.fit(third_layer_code, Y_train,
                epochs = 8, batch_size = 512,
                validation_split = 0.25,
                shuffle = True)

# Setting up the weights of the not-as-deep autoencoder
nad_deep_autoencoder.layers[1].set_weights(autoencoder1.layers[1].get_weights()) # first dense layer
nad_deep_autoencoder.layers[2].set_weights(autoencoder1.layers[2].get_weights()) # first bn layer
nad_deep_autoencoder.layers[3].set_weights(autoencoder2.layers[1].get_weights()) # second dense layer
nad_deep_autoencoder.layers[4].set_weights(autoencoder2.layers[2].get_weights()) # second bn layer
nad_deep_autoencoder.layers[5].set_weights(autoencoder3.layers[1].get_weights()) # third dense layer
nad_deep_autoencoder.layers[6].set_weights(autoencoder3.layers[2].get_weights()) # third bn layer
nad_deep_autoencoder.layers[7].set_weights(softmax1.layers[1].get_weights()) # fourth dense layer

val_preds = nad_deep_autoencoder.predict(X_test)
#val_preds=np.round(val_preds)

######################################
for j in range(len(val_preds)):
    for i in range(15*3):
        if val_preds[j][2*i]>val_preds[j][(2*i)+1]:
            val_preds[j][2*i]=1
            val_preds[j][(2*i)+1]=0
        else:
            val_preds[j][2 * i] = 0
            val_preds[j][(2 * i) + 1] = 1

n_correct = np.sum(np.equal(val_preds, Y_test).astype(int))
total = float(len(val_preds))
print("Test Accuracy:", n_correct / total)

nad_deep_autoencoder.fit(X_train, Y_train,
               epochs=20, batch_size=500,
               validation_split=0.25,
               shuffle=True)

val_preds = nad_deep_autoencoder.predict(X_test)
print(val_preds[0])
#val_preds=np.round(val_preds)

######################################
for j in range(len(val_preds)):
    for i in range(15*3):
        if val_preds[j][2*i]>val_preds[j][(2*i)+1]:
            val_preds[j][2*i]=1
            val_preds[j][(2*i)+1]=0
        else:
            val_preds[j][2 * i] = 0
            val_preds[j][(2 * i) + 1] = 1


n_correct = np.sum(np.equal(val_preds, Y_test).astype(int))
total = float(len(val_preds))
print("Test Accuracy:", n_correct / total)

#print(np.round(val_preds[0]),Y_test[0])