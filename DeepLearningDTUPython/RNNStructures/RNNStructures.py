#RNN from the weekly exercises.
from __future__ import absolute_import, division, print_function

#%matplotlib inline
# %matplotlib nbagg
import tensorflow as tf
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from data_generator_tensorflow import get_batch, print_valid_characters
from DataGenerationCommaPlacement import get_batch_comma
from DataGenerationCommaPlacement import num_of_training_samples

import os
import sys
sys.path.append(os.path.join('.', '..')) 
from tempUtils import utils 

import tf_utils


# At the bottom of the script there is some code which saves the model.
# If you wish to restore your model from a previous state use this function.
load_model = False

#-------------------------------------------- play around with input data... to be replaced with specific code------------------
batch_size = 2
#inputs, inputs_seqlen, targets_in, targets_out, targets_seqlen, targets_mask, \
#text_inputs, text_targets_in, text_targets_out = \
#        get_batch(batch_size=batch_size, max_digits=10, min_digits=5)

inputs, inputs_seqlen, targets_in, targets_out, targets_in_seqlen, targets_out_seqlen, targets_mask, \
text_targets_in, text_targets_out = \
        get_batch_comma(batch_size=batch_size, indices_of_interest = [0,1])
    
    
print(inputs.shape)
print("input types:", inputs.dtype)#, inputs_seqlen.dtype, targets_in.dtype, targets_out.dtype, targets_seqlen.dtype)

for i in range(batch_size):
    print("\nSAMPLE",i)
    print("ENCODED INPUTS:\t\t\t", inputs[i])
    print("INPUTS SEQUENCE LENGTH:\t\t", inputs_seqlen[i])
    print("TEXT TARGETS OUTPUT:\t\t", text_targets_out[i])
    print("TEXT TARGETS INPUT:\t\t", text_targets_in[i])
    print("ENCODED TARGETS OUTPUT:\t\t", targets_out[i])
    print("ENCODED TARGETS INPUT:\t\t", targets_in[i])
    print("INPUT TARGETS SEQUENCE LENGTH:\t", targets_in_seqlen[i])
    print("OUTPUT TARGETS SEQUENCE LENGTH:\t", targets_out_seqlen[i])
    print("TARGETS MASK:\t\t\t", targets_mask[i])
#---------------------------------------------------setting up the model-----------------------------------------------------
# resetting the graph
tf.reset_default_graph()

# Setting up hyperparameters and general configs
NUM_INPUTS = 125002
NUM_OUTPUTS = 20 #(0-19 +",")

BATCH_SIZE = 100
# try various learning rates 1e-2 to 1e-5
LEARNING_RATE = 0.005 #0.005
X_EMBEDDINGS = 8
t_EMBEDDINGS = 8
NUM_UNITS_ENC = 128
NUM_UNITS_DEC = 128
number_of_layers = 4


# Setting up placeholders, these are the tensors that we "feed" to our network
Xs = tf.placeholder(tf.int32, shape=[None, None], name='X_input')
ts_in = tf.placeholder(tf.int32, shape=[None, None], name='t_input_in')
ts_out = tf.placeholder(tf.int32, shape=[None, None], name='t_input_out')
X_len = tf.placeholder(tf.int32, shape=[None], name='X_len')
t_in_len = tf.placeholder(tf.int32, shape=[None], name='t_in_len')
t_out_len = tf.placeholder(tf.int32, shape=[None], name='t_out_len')
t_mask = tf.placeholder(tf.float32, shape=[None, None], name='t_mask')


### Building the model
# first we build the embeddings to make our characters into dense, trainable vectors
X_embeddings = tf.get_variable('X_embeddings', [NUM_INPUTS, X_EMBEDDINGS],
                               initializer=tf.random_normal_initializer(stddev=0.1))
t_embeddings = tf.get_variable('t_embeddings', [NUM_INPUTS, t_EMBEDDINGS],
                               initializer=tf.random_normal_initializer(stddev=0.1))

X_embedded = tf.gather(X_embeddings, Xs, name='embed_X')
t_embedded = tf.gather(t_embeddings, ts_in, name='embed_t')


## forward encoding - use deep cell, with multiple GRU cells stacked on top of each other.
enc_cell = tf.nn.rnn_cell.GRUCell(NUM_UNITS_ENC)

def gru_cell():
    return tf.nn.rnn_cell.GRUCell(NUM_UNITS_ENC)
stacked_gru = tf.contrib.rnn.MultiRNNCell(
    [gru_cell() for _ in range(number_of_layers)])


enc_out, enc_state = tf.nn.dynamic_rnn(cell=stacked_gru, inputs=X_embedded,
                                  sequence_length=X_len, dtype=tf.float32)
# dynamic Rnn of a stack, outputs the final state of each layer. We are only interested in the final state of the final layer
enc_state = enc_state[-1]

# use below incase TF's makes issues
# enc_state, _ = tf_utils.encoder(X_embedded, X_len, 'encoder', NUM_UNITS_ENC)
#
# enc_state = tf.concat(1, [enc_state, enc_state])

## decoding
# note that we are using a wrapper for decoding here, this wrapper is hardcoded to only use GRU
# check out tf_utils to see how you make your own decoder

# setting up weights for computing the final output
W_out = tf.get_variable('W_out', [NUM_UNITS_DEC, NUM_OUTPUTS])
b_out = tf.get_variable('b_out', [NUM_OUTPUTS])

dec_out, valid_dec_out = tf_utils.decoder(enc_state, t_embedded, t_in_len, 
                                          NUM_UNITS_DEC, t_embeddings,
                                          W_out, b_out)

#This code extracts the output of just the last rnn cell in the sequence. Enc_out has the format 
#(Batch_size, max_sequence_length, final state)
#out_tensor = tf.slice(enc_out, [0, 0 ,0],[-1,1,-1])
#valid_out_tensor = tf.slice(enc_out,[0,0,0],[-1,1,-1])

## reshaping to have [batch_size*seqlen, num_units]
out_tensor = tf.reshape(dec_out, [-1, NUM_UNITS_DEC])
valid_out_tensor = tf.reshape(valid_dec_out, [-1, NUM_UNITS_DEC])

# computing output
out_tensor = tf.matmul(out_tensor, W_out) + b_out
valid_out_tensor = tf.matmul(valid_out_tensor, W_out) + b_out

#out_tensor = tf.expand_dims(out_tensor, 1)
#valid_out_tensor = tf.expand_dims(valid_out_tensor, 1)

## reshaping back to sequence
# print('X_len', tf.shape(X_len)[0])
b_size = tf.shape(X_len)[0] # use a variable we know has batch_size in [0]
num_out = tf.constant(NUM_OUTPUTS) # casting NUM_OUTPUTS to a tensor variable
out_shape = tf.concat([tf.expand_dims(b_size, 0),
                      tf.expand_dims(t_out_len[0],0),
                      tf.expand_dims(num_out, 0)],
                     axis=0)

out_tensor = tf.reshape(out_tensor, out_shape)
valid_out_tensor = tf.reshape(valid_out_tensor, out_shape)
# handling shape loss
#out_tensor.set_shape([None, None, NUM_OUTPUTS])
y = out_tensor
y_valid = valid_out_tensor

# print all the variable names and shapes
for var in tf.global_variables ():
    s = var.name + " "*(40-len(var.name))
    print(s, var.value().get_shape())

#--------------------------------------------Defining loss, cost funtion etc.----------------------------------------------------

def loss_and_acc(preds):
    # sequence_loss_tensor is a modification of TensorFlow's own sequence_to_sequence_loss
    # TensorFlow's seq2seq loss works with a 2D list instead of a 3D tensors
    loss = tf_utils.sequence_loss_tensor(preds, ts_out, t_mask, NUM_OUTPUTS) # notice that we use ts_out here!

    ## if you want regularization
    #reg_scale = 0.00001
    #regularize = tf.contrib.layers.l2_regularizer(reg_scale)
    #params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    #reg_term = sum([regularize(param) for param in params])
    #loss += reg_term
    
    ## calculate accuracy
    argmax = tf.to_int32(tf.argmax(preds, 2))
    correct = tf.to_float(tf.equal(argmax, ts_out)) * t_mask
    accuracy = tf.reduce_sum(correct) / tf.reduce_sum(t_mask)
    return loss, accuracy, argmax

loss, accuracy, predictions = loss_and_acc(y)
loss_valid, accuracy_valid, predictions_valid = loss_and_acc(y_valid)

# use lobal step to keep track of our iterations
global_step = tf.Variable(0, name='global_step', trainable=False)

# pick optimizer, try momentum or adadelta
optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

# extract gradients for each variable
grads_and_vars = optimizer.compute_gradients(loss)

## add below for clipping by norm
# gradients, variables = zip(*grads_and_vars)  # unzip list of tuples
# clipped_gradients, global_norm = (
#    tf.clip_by_global_norm(gradients, self.clip_norm) )
# grads_and_vars = zip(clipped_gradients, variables)
# apply gradients and make trainable function
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

# print all the variable names and shapes
# notice that we now have the optimizer Adam as well!
for var in tf.all_variables():
    s = var.name + " "*(40-len(var.name))
    print(s, var.value().get_shape())

#----------------------------------------------------Test the forward pass:-----------------------------------------------------
## Start the session
# restricting memory usage, TensorFlow is greedy and will use all memory otherwise
gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.35)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts))

# Initialize parameters
if load_model:
    try:
        tf.train.Saver().restore(sess, "/save/model.ckpt")
    except:
        sess.run(tf.global_variables_initializer())
        print('Model not found, new parameters initialized')
else:
    sess.run(tf.global_variables_initializer())

# as always, test the forward pass and initialize the tf.Session!
for i in range(batch_size):
    print("\nSAMPLE",i)
   # print("TEXT INPUTS:\t\t\t", text_inputs[i])
    print("TEXT TARGETS INPUT:\t\t", text_targets_in[i])

feed_dict = {Xs: inputs, X_len: inputs_seqlen, ts_in: targets_in,
             ts_out: targets_out, t_in_len: targets_in_seqlen, t_out_len: targets_out_seqlen}

# test training forwardpass
print("targets seqlen", targets_in_seqlen)
print("targets out seqlen", targets_out_seqlen)
fetches = [y, X_embedded, enc_state]
#fetches = [dec_out]
res = sess.run(fetches=fetches, feed_dict=feed_dict)
print("out shape:", res[0].shape)
print("inputs", inputs)
print("input length", inputs_seqlen)
print("encoder state", res[2].shape)
print("targets output", targets_out)

print("y", res[0].shape)
print("X_embedded", res[1].shape)

# test validation forwardpass
fetches = [y_valid]
res = sess.run(fetches=fetches, feed_dict=feed_dict)
print("y_valid", res[0].shape)

#Some Data hyperparameters
num_epochs = 10
num_of_training_samples_loaded = num_of_training_samples()
num_of_samples_for_validation = 1000
num_of_training_samples_minus_validation = num_of_training_samples_loaded - num_of_samples_for_validation
num_batches_train = int(num_of_training_samples_minus_validation / BATCH_SIZE)
print(num_batches_train)

#--------------------------------------------------Generate validation  data - to be replaced--------------------------------------
#Generate some validation data
validation_indices = list(range(num_of_training_samples_minus_validation, num_of_training_samples_minus_validation+num_of_samples_for_validation))
print(len(validation_indices))
X_val, X_len_val, t_in_val, t_out_val, t_len_in_val, t_len_out_val,t_mask_val, \
text_targets_in_val, text_targets_out_val = \
    get_batch_comma(batch_size=num_of_samples_for_validation, indices_of_interest = validation_indices)
print("X_val", X_val.shape)
print("t_out_val", t_out_val.shape)

#--------------------------------------------------Training ----------------------------------------------------------------------

#%%time
## If you get an error, remove this line! It makes the error message hard to understand.

# setting up running parameters
val_interval = 200

samples_val = []
costs, accs_val = [], []
plt.figure()
batch_number = 0
try:
    for epoch in range(num_epochs):
        epoch_batches = []
        accs_val = []
        print("Epoch: ", epoch)
        for i in range(num_batches_train): 
            # load data
            #print("Epoch {0} new Batch: {1} ".format(epoch, i))
            
            #Select random part of the training data   
            idxs = np.random.choice(range(num_of_training_samples_minus_validation), size=(BATCH_SIZE), replace=False)
            X_tr, X_len_tr, t_in_tr, t_out_tr, t_len_in_tr, t_len_out_tr, t_mask_tr, \
            text_targets_in_tr, text_targets_out_tr = \
                get_batch_comma(batch_size=BATCH_SIZE, indices_of_interest = idxs)
            # make fetches
            fetches_tr = [train_op, loss, accuracy]
            # set up feed dict
            feed_dict_tr = {Xs: X_tr, X_len: X_len_tr, ts_in: t_in_tr,
                 ts_out: t_out_tr, t_in_len: t_len_in_tr, t_out_len: t_len_out_tr, t_mask: t_mask_tr}
            # run the model
            res = tuple(sess.run(fetches=fetches_tr, feed_dict=feed_dict_tr))
            _, batch_cost, batch_acc = res
            costs += [batch_cost]
            
            #if samples_processed % 1000 == 0: print(batch_cost, batch_acc)
            #validation data
            if i % val_interval == 0:
                print("validating")
                fetches_val = [accuracy_valid, y_valid]
                feed_dict_val = {Xs: X_val, X_len: X_len_val, ts_in: t_in_val,
                 ts_out: t_out_val, t_in_len: t_len_in_val, t_out_len: t_len_out_val, t_mask: t_mask_val}
                res = tuple(sess.run(fetches=fetches_val, feed_dict=feed_dict_val))
                
                acc_val, output_val = res
                
                accs_val += [acc_val]
                epoch_batches +=[epoch*num_batches_train+i]
                print("Epoch-batches ", epoch_batches)
                print("accs_val: ", accs_val)
                print("accs_train: ", accuracy)
                plt.plot(epoch_batches, accs_val, 'g-')
                plt.ylabel('Validation Accuracy', fontsize=15)
                plt.xlabel('Epoch+Sample', fontsize=15)
                plt.title('', fontsize=20)
                plt.grid('on')
                plt.savefig("out.png")
                display.display(display.Image(filename="out.png"))
                display.clear_output(wait=True)
except KeyboardInterrupt:
    pass

print('Done')
#-------------------------------------------------------------------plotting-----------------------------------------------------
#plot of validation accuracy for each target position
plt.figure(figsize=(7,7))
plt.plot(np.mean(np.argmax(output_val,axis=2)==t_out_val,axis=0))
plt.ylabel('Accuracy', fontsize=15)
plt.xlabel('Target position', fontsize=15)
#plt.title('', fontsize=20)
plt.grid('on')
plt.show()
#why do the plot look like this?
#-------------------------------------------------------------------saving the model----------------------------------------------
## Save model
# Read more about saving and loading models at https://www.tensorflow.org/programmers_guide/saved_model

# Save model
save_path = tf.train.Saver().save(sess, "/tmp/model.ckpt")
print("Model saved in file: %s" % save_path)

## Close the session, and free the resources
sess.close()