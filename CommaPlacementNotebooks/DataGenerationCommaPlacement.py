from __future__ import absolute_import, division, print_function 
import csv
import sys
import re
import numpy as np

# Quick and dirty fix for too large csv file
maxInt = sys.maxsize
decrement = True

while decrement:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True


#with open("Data\word_stems.csv", mode='r') as wordStems:
#    reader = csv.reader(wordStems, delimiter = ";")
#    encoding_to_word_stem = {rows[1]:rows[0] for rows in reader}

#print("Wordstem", encoding_to_word_stem['99060'])




# load the dictionary of word stem into memory

# load a number of sentences, encoded as a tuple of 2 numbers.
with open("Data\onehot_encoded.csv", mode='r') as oneHotEncoded:    
    reader = csv.reader(oneHotEncoded, delimiter = ";")
    listOfSentences = []
    count = 0
    for row in reader:
        #if count>20000:
        #    break
        #count =count+1
        mads = ""
        for noget in row:
            mads=mads+noget
        pattern = re.compile(r'\(([0-9]+),[0-9]+\)')
        s = ""
        for number in re.findall(pattern, mads):
            s = s + str(number) + " "
        sen = len(s.split(' '))
        if "99060" in s and sen<20:  
            #print(s)
            listOfSentences.append(s)
        
print("sentence number 6: ", len(listOfSentences))
stop_character = start_character = '#'

def num_of_training_samples():
    return len(listOfSentences)

def get_batch_comma(batch_size = 100, indices_of_interest = [0,1,2]):
    
    if batch_size != len(indices_of_interest):
        return #failure - figure out how to throw error message
    text_inputs = []
    int_inputs = []
    text_targets_in = []
    text_targets_out = []
    int_targets_in = []
    int_targets_out = []
    #first encode all the input and targets
    for i in indices_of_interest:
        #find length of sequence:
        tar_len = len(listOfSentences[i])
        
        text_target = inp_str = listOfSentences[i]
        listOfIntTarget = text_target.split(' ')
        #print("input as list:", listOfIntTarget)        

        text_target_in = start_character + text_target
        
        text_target_out = str(listOfIntTarget.index('99060'))+stop_character #text_target[0] + stop_character
        #print("Target: ", text_target_out)
                
        #generate the targets as a list of integers
        int_target_in = [int(digit) for digit in listOfIntTarget if digit is not '']

        int_target_out = [listOfIntTarget.index('99060')]
        #remove the comma from the sentence, as the task is to predict the comma, and not just look for it
        #but leave the comma in the text input, as that is just used for easier debugging
        int_target_in.remove(99060)
        
        #print("int_target_in", int_target_in)
        #print(int_target_out)

        text_targets_in.append(text_target_in)
        text_targets_out.append(text_target_out)
        int_inputs.append(int_target_in)
        int_targets_in.append(int_target_in)
        int_targets_out.append(int_target_out)        
  
    #determine padding of sequence:
    max_input_len = max([len(i) for i in int_inputs])
    inputs = np.zeros((batch_size, max_input_len))
#    input_masks = np.zeros((batch_size,max_input_len))
    for (i,inp) in enumerate(int_inputs):
        cur_len = len(list(inp))
        inputs[i,:cur_len] = inp
#        input_masks[i,:cur_len] = 1
#     inputs_seqlen = np.asarray(map(len, int_inputs))
    inputs_seqlen = np.asarray([len(i) for i in int_targets_in])
    
    #pad the target sequene of the encoding 
    max_target_in_len = max([len(list(i)) for i in int_targets_in])
    targets_in = np.zeros((batch_size, max_target_in_len))
    for (i, tar) in enumerate(int_targets_in):
        cur_len = len(tar)
        targets_in[i, :cur_len] = tar
#     targets_seqlen = np.asarray(map(len, int_targets_in))
    targets_in_seqlen = np.asarray([len(i) for i in int_targets_in])

    #Determine the length of the target sequence of the *decoding*
    max_target_out_len = max(map(len, int_targets_out))
    targets_mask = np.zeros((batch_size, max_target_out_len))
    targets_out = np.zeros((batch_size, max_target_out_len))
    for (i,tar) in enumerate(int_targets_out):
        cur_len = len(tar)
        targets_out[i,:cur_len] = tar
        targets_mask[i,:cur_len] = 1    
    targets_out_seqlen = np.asarray([1 for i in int_targets_out])
    
        # The encoded inputs
        # The length of the input sequence
        #The encoded input sequence (same as the inputs, as no encoding happens in the 'encoder network'
        # the target of the output sequence. I.e. the target of the network
        #the length of the target output sequence
        #the important parts of the output sequence
    return  inputs.astype('int32'),  \
            inputs_seqlen.astype('int32'), \
            targets_in.astype('int32'), \
            targets_out.astype('int32'), \
            targets_in_seqlen.astype('int32'), \
            targets_out_seqlen.astype('int32'),\
            targets_mask.astype('float32'), \
            text_targets_in, \
            text_targets_out
           #int_target_in, \
           #int_target_out, \
                    
get_batch_comma(2, [0,1])
# convert these numbers as a single vector. 
class class1(object):
    """description of class"""


