#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import pretty_midi
import glob
import collections
import pickle
import random

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from tensorflow.data import Dataset
import IPython.display

from tqdm import tqdm
from music21 import converter, instrument, note, chord, stream


# ## Usefull methods

# ### Method to create a midi file

# In[2]:


def createMidi(combined_notes, fileName):
    offset = 0
    output_notes = []
    
    for combine in combined_notes:

        for midi_notes in ind_to_char[combine].split('&'):
            try:
                new_note = note.Note(float(midi_notes))
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
                offset += 0.5
            except:
                pass
            
    midi_stream = stream.Stream(output_notes)
    
    print('Saving Output file as midi....')
    print(len(output_notes))

    midi_stream.write('midi', fp=fileName)


# In[3]:


objects = []
with (open("MuseData.pickle", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break


# In[4]:


all_notes = []
for song in objects[0]['test']:
    for combineNote in song:
        notes_string = ''
        while len(combineNote) > 3:
            combineNote.pop(random.randrange(len(combineNote)))
        for notes in combineNote:
            notes_string += str(notes) + '&'
        all_notes.append(notes_string[:-1])
        
for song in objects[0]['train']:
    for combineNote in song:
        notes_string = ''
        while len(combineNote) > 3:
            combineNote.pop(random.randrange(len(combineNote)))
        for notes in combineNote:
            notes_string += str(notes) + '&'
        all_notes.append(notes_string[:-1])
        
for song in objects[0]['valid']:
    for combineNote in song:
        notes_string = ''
        while len(combineNote) > 3:
            combineNote.pop(random.randrange(len(combineNote)))
        for notes in combineNote:
            notes_string += str(notes) + '&'
        all_notes.append(notes_string[:-1])
        


# In[5]:


len(all_notes)


# ### create a diccionary of notes and encode the full pack of notes

# In[6]:


diccionary = sorted(set(item for item in all_notes))

char_to_ind = {u:i for i, u in enumerate(diccionary)}

ind_to_char = np.array(diccionary)

encoded_text = np.array([char_to_ind[c] for c in all_notes])


# In[7]:


len(diccionary)


# ## Creating batches

# ### calculate total num of sequences

# In[8]:


seq_len = 50

total_num_seq = len(all_notes)//(seq_len+1)
total_num_seq


# ### divide the entire dataset of notes into sequences

# In[9]:


import tensorflow as tf


# In[10]:


char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)

sequences = char_dataset.batch(seq_len+1, drop_remainder=True)

def create_seq_targets(seq):
    input_txt = seq[:-1]
    target_txt = seq[1:]
    return input_txt, target_txt

dataset = sequences.map(create_seq_targets)


# ### create batched

# In[11]:


batch_size = 50

buffer_size = 10000

dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
dataset


# ## Create the model

# In[12]:


vocab_size = len(diccionary)

# The embedding dimension. Your choice.
embed_dim = 128

# Number of RNN units. Your choice. YOU MUST EXPERIMENT WITH THIS NUMBER.
rnn_neurons = 256

# Value of the Dropout
dropout_value = 0.2


# In[13]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout,GRU

from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.losses import sparse_categorical_crossentropy


# In[14]:


def sparse_cat_loss(y_true,y_pred):
    return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

def create_model(vocab_size, embed_dim, rnn_neurons, batch_size):

    model = Sequential()
    model.add(Embedding(input_dim = vocab_size, output_dim = embed_dim, batch_input_shape=[batch_size, None]))
    model.add(LSTM(rnn_neurons,
                   return_sequences=True,
                   stateful=True,
                   recurrent_initializer=GlorotNormal()
                  ))
    model.add(Dropout(dropout_value))
    model.add(LSTM(rnn_neurons,
                   return_sequences=True,
                   stateful=True,
                   recurrent_initializer=GlorotNormal()
                  ))
    model.add(Dropout(dropout_value))
    model.add(Dense(vocab_size))
    return model
 
# create your model here, passing the parameters
model = create_model(vocab_size, embed_dim, rnn_neurons, batch_size)


# In[15]:


model.summary()


# ## Generate note using the model without training

# ### expecting to see something random

# In[16]:


for input_example_batch, target_example_batch in dataset.take(1):

    example_batch_predictions = model(input_example_batch)

    print(example_batch_predictions.shape, " <=== (batch_size, sequence_length, vocab_size)")


example_batch_predictions

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices

# Reformat to not be a lists of lists
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
sampled_indices

createMidi(input_example_batch[0], 'input_piano_seq.mid')
createMidi(sampled_indices, 'sample_piano_seq.mid')


# ### Compile and train the model

# In[17]:


from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy 


# In[19]:


model.compile(loss = sparse_cat_loss, optimizer = Adam(learning_rate=0.005), metrics=['accuracy'])

es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience = 10)

r = model.fit(dataset, epochs = 10, callbacks=[es])


# In[23]:


plt.plot(r.history['loss'], label='loss')

plt.title('Training error on Piano model')
plt.xlabel('epoch')
plt.ylabel('error')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='accuracy')

plt.title('Training accuracy on Piano model')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# In[21]:


model_name = 'piano_gen2_.h5'
model.save(model_name ,save_format='h5')


# ## Generate music using the trained model

# In[22]:


from tensorflow.keras.models import load_model

model = create_model(vocab_size, embed_dim, rnn_neurons, batch_size=1)
model.load_weights(model_name)
model.build(tf.TensorShape([1, None]))
model.summary()

def generate_text(model, start_seed,gen_size=100,temp=1):
      
    # Number of characters to generate
    num_generate = gen_size

    # Vecotrizing starting seed text
    input_eval = [char_to_ind[s] for s in start_seed]

    # Expand to match batch format shape
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty list to hold resulting generated text
    text_generated = []

    # Temperature effects randomness in our resulting text
    # The term is derived from entropy/thermodynamics.
    # The temperature is used to effect probability of next characters.
    # Higher probability == lesss surprising/ more expected
    # Lower temperature == more surprising / less expected
 
    temperature = temp

    # Here batch size == 1
    model.reset_states()

    for i in range(num_generate):

      # Generate Predictions
        predictions = model(input_eval)

      # Remove the batch shape dimension
        predictions = tf.squeeze(predictions, 0)

      # Use a cateogircal disitribution to select the next character
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # Pass the predicted charracter for the next input
        input_eval = tf.expand_dims([predicted_id], 0)

      # Transform back to character letter
        text_generated.append(predicted_id)
    
    createMidi(text_generated, 'finalPiano.mid')

    return 'Music generated'

print(generate_text(model,all_notes[:50],gen_size=200))

