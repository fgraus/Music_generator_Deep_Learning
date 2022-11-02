#!/usr/bin/env python
# coding: utf-8

# # Bach generator

# In[2]:


import pathlib
import pretty_midi
import glob
import collections

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from tensorflow.data import Dataset
import IPython.display

from tqdm import tqdm
from music21 import converter, instrument, note, stream


# ## Usefull methods

# ### Method to create a midi file

# In[3]:


def createMidi(notes, fileName):
    offset = 0
    output_notes = []
    
    for nota in notes:
        translated_note = ind_to_char[nota]
        new_note = note.Note(int(translated_note.split('&')[0]))
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)
        
        try:
            offset += float(translated_note.split('&')[1])
        except:
            offset += float(translated_note.split('&')[1].split('/')[0]) / float(translated_note.split('&')[1].split('/')[1])
        
    
    midi_stream = stream.Stream(output_notes)
    
    print('Saving Output file as midi....')

    midi_stream.write('midi', fp=fileName)


# ## Handle data

# ### get file names

# In[4]:


data_dir = pathlib.Path('data/bach')
filenames = glob.glob(str(data_dir/'*.mid*'))
print('Number of files: ', len(filenames))


# ### extract all the notes from the files (TAKES A LOT OF TIME)

# In[ ]:


all_notes = []
for i in tqdm(range(len(filenames))):
    file = filenames[i]
    midi = converter.parse(file)

    flatten_notes = midi.flat.notes

    for flat_note in flatten_notes:
        if isinstance(flat_note, note.Note):
            all_notes.append(str(flat_note.pitch.midi)+'&'+str(flat_note.duration.quarterLength))


# In[17]:


len(all_notes)


# In[ ]:


get_ipython().run_line_magic('store', 'all_notes')


# In[19]:


get_ipython().run_line_magic('store', '-r all_notes')


# ### plot histograph frecuencies

# In[20]:


all_notes[:10]


# In[8]:


def noteToFreq(note):
    a = 440 #frequency of A (coomon value is 440Hz)
    return (a / 32) * (2 ** ((note - 9) / 12))


# In[10]:


freq = []
for no in all_notes:
    freq.append(noteToFreq(int(no.split('&')[0])))


# In[16]:


df = pd.DataFrame(freq, columns=['Frecuency'])
df.hist()


# ### create a diccionary of notes and encode the full pack of notes

# In[21]:


diccionary = sorted(set(item for item in all_notes))

char_to_ind = {u:i for i, u in enumerate(diccionary)}

ind_to_char = np.array(diccionary)

encoded_text = np.array([char_to_ind[c] for c in all_notes])


# In[22]:


len(diccionary)


# ## Creating batches

# ### calculate total num of sequences

# In[ ]:


seq_len = 50

total_num_seq = len(all_notes)//(seq_len+1)
total_num_seq


# ### divide the entire dataset of notes into sequences

# In[ ]:


import tensorflow as tf


# In[ ]:


char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)

sequences = char_dataset.batch(seq_len+1, drop_remainder=True)

def create_seq_targets(seq):
    input_txt = seq[:-1]
    target_txt = seq[1:]
    return input_txt, target_txt

dataset = sequences.map(create_seq_targets)


# ### create batched

# In[ ]:


batch_size = 50

buffer_size = 10000

dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
dataset


# ## Create the model

# In[ ]:


vocab_size = len(diccionary)

# The embedding dimension. Your choice.
embed_dim = 128

# Number of RNN units. Your choice. YOU MUST EXPERIMENT WITH THIS NUMBER.
rnn_neurons = 256

# Value of the Dropout
dropout_value = 0.2


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout,GRU

from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.losses import sparse_categorical_crossentropy


# In[ ]:


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
 
model = create_model(vocab_size, embed_dim, rnn_neurons, batch_size)


# In[ ]:


model.summary()


# ## Generate note using the model without training

# ### expecting to see something random

# In[ ]:


for input_example_batch, target_example_batch in dataset.take(1):

    example_batch_predictions = model(input_example_batch)

    print(example_batch_predictions.shape, " <=== (batch_size, sequence_length, vocab_size)")


example_batch_predictions

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices

# Reformat to not be a lists of lists
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
sampled_indices

createMidi(input_example_batch[0], 'input_bach_seq.mid')
createMidi(sampled_indices, 'sample_bach_seq.mid')


# ### Compile and train the model

# In[ ]:


from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy 


# In[ ]:


model.compile(loss = sparse_cat_loss, optimizer = Adam(learning_rate=0.01), metrics=['accuracy'])

es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience = 10)

r = model.fit(dataset, epochs = 50, callbacks=[es])


# In[ ]:


plt.plot(r.history['loss'], label='loss')

plt.title('Training error on Bach model')
plt.xlabel('epoch')
plt.ylabel('error')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='accuracy')

plt.title('Training accuracy on Bach model')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# In[ ]:


model_name = 'bach_gen_.h5'
model.save(model_name ,save_format='h5')


# ## Generate music using the trained model

# In[ ]:


from tensorflow.keras.models import load_model

model = create_model(vocab_size, embed_dim, rnn_neurons, batch_size=1)
#model.load_weights(model_name)
model.build(tf.TensorShape([1, None]))
model.summary()

def generate_text(model, start_seed,gen_size=100,temp=0.25):
      
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
    
    createMidi(text_generated, 'finalBach.mid')

    return 'Music generated'

print(generate_text(model,all_notes[:50],gen_size=200))


# In[ ]:





# In[ ]:




