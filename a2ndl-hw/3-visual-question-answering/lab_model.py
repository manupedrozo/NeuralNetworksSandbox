import os
import tensorflow as tf
import numpy as np
from datetime import datetime

# Set the seed for random operations. 
# This let our experiments to be reproducible. 
SEED = 1234
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Get current working directory
cwd = os.getcwd()

# Given the text we will:
# 1: divide the text into subsequences of N chars
# 2: build (input, target) pairs, where input is a sequence and target is the 
#    same sequence shifted of 1 char
# 3: train a recurrent neural network to predict the next char after each input char
# 4: use the learned model to generate text

# -------------------------- DATASET --------------------------
# Prepare dataset
# ---------------

# Read full text
with open(os.path.join('lab/Cantico_di_Natale.txt'), 'r') as f:
    full_text = f.read()

full_text_length = len(full_text)
print('Full text length:', full_text_length)

# Create vocabulary
vocabulary = sorted(list(set(full_text)))

print('Number of unique characters:', len(vocabulary))
print(vocabulary)

# Dictionaries for char-to-int/int-to-char conversion
ctoi = {c:i for i, c in enumerate(vocabulary)}
itoc = {i:c for i, c in enumerate(vocabulary)}

# Create input-target pairs
# e.g., given an input sequence 
# 'Hell' predict the next characters 'ello'
# Thus,
# extract from the full text sequences of length seq_length as x and 
# the corresponding seq_length+1 character as target

# Define number of characters to be considered for the prediction
seq_length = 100

X = [] # will contain all the sequences 
Y = [] # will contain for each sequence in X the next characters
X_enc = []
Y_enc = []
# Cycle over the full text
step = 1 
for i in range(0, full_text_length - (seq_length), step):
    sequence = full_text[i:i+seq_length]
    target = full_text[i+1:i+seq_length+1]
    X.append(sequence)
    Y.append(target)
    X_enc.append([ctoi[c] for c in sequence])
    Y_enc.append([ctoi[c] for c in target])
    
X = np.array(X)
Y = np.array(Y)
X_enc = np.array(X_enc)
Y_enc = np.array(Y_enc)
    
print('Number of sequences in the dataset:', len(X))

print("Input Sequence: {}".format(X[0]))
print("Target Sequence: {}".format(Y[0]))

# Create data loaders
# -------------------

# Batch size
bs = 256

# Encode characters. Many ways, for example one-hot encoding.
def char_encode(x_, y_):
    return tf.one_hot(x_, len(vocabulary)), tf.one_hot(y_, len(vocabulary))

# Prepare input x to match recurrent layer input shape 
# -> (bs, seq_length, input_size)

# Training
train_dataset = tf.data.Dataset.from_tensor_slices((X_enc, Y_enc))
train_dataset = train_dataset.shuffle(buffer_size=X_enc.shape[0])
train_dataset = train_dataset.map(char_encode)
train_dataset = train_dataset.batch(bs)
train_dataset = train_dataset.repeat()

# -------------------------- MODEL --------------------------

# Build Recurrent Neural Network
# ------------------------------

# We build two models, one for training and one for inference. The two models 
# SHARE THE WEIGHTS, thus the inference model will use the learned weights after training

# Training and inference model differ only for the initialization of the state 
# in the lstm. In the inference model the state of the lstm is obtained through 
# input layers. In this way, we can provide the prediction at time t-1 as input 
# at time t for the generation of text at inference time.

# Model architecture (2 stacked lstm layers): Input -> LSTM-1 -> LSTM-2 -> Dense 

# Hidden size (state)
h_size = 128

# Model
input_x = tf.keras.Input(shape=(None, len(vocabulary)))

lstm1 = tf.keras.layers.LSTM(
    units=h_size, batch_input_shape=[None, None, len(vocabulary)], 
    return_sequences=True, return_state=True, stateful=False)
lstm2 = tf.keras.layers.LSTM(
    units=h_size, return_sequences=True, 
    return_state=True, stateful=False)
dense = tf.keras.layers.Dense(units=len(vocabulary), activation='softmax')

x, _, _ = lstm1(input_x)
x, _, _ = lstm2(x)
out = dense(x)

train_model = tf.keras.Model(
    inputs=input_x, outputs=out)

# Inference Model
h1_in = tf.keras.Input(shape=[h_size])
c1_in = tf.keras.Input(shape=[h_size])
h2_in = tf.keras.Input(shape=[h_size])
c2_in = tf.keras.Input(shape=[h_size])

x, h1, c1 = lstm1(input_x, initial_state=[h1_in, c1_in])
x, h2, c2 = lstm2(x, initial_state=[h2_in, c2_in])
out = dense(x)

inference_model = tf.keras.Model(
    inputs=[input_x, h1_in, c1_in, h2_in, c2_in], 
    outputs=[out, h1, c1, h2, c2])

train_model.summary()

inference_model.summary()
#model.weights

# Optimization params
# -------------------

# Loss
loss = tf.keras.losses.CategoricalCrossentropy()

# learning rate
lr = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
# -------------------

# Validation metrics
# ------------------

metrics = ['accuracy']
# ------------------

# Compile Model
train_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

cwd = os.getcwd()

exps_dir = os.path.join('lab', 'text_gen_experiments')
if not os.path.exists(exps_dir):
    os.makedirs(exps_dir)

now = datetime.now().strftime('%b%d_%H-%M-%S')

exp_name = 'exp'

exp_dir = os.path.join(exps_dir, exp_name + '_' + str(now))
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
    
callbacks = []

# Model checkpoint
# ----------------
ckpt_dir = os.path.join(exp_dir, 'ckpts')
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp_{epoch:02d}.ckpt'), 
                                                   save_weights_only=True)  # False to save the model directly
callbacks.append(ckpt_callback)

# ----------------

# Visualize Learning on Tensorboard
# ---------------------------------
tb_dir = os.path.join(exp_dir, 'tb_logs')
if not os.path.exists(tb_dir):
    os.makedirs(tb_dir)
    
# By default shows losses and metrics for both training and validation
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                             profile_batch=0,
                                             histogram_freq=1)  # if 1 shows weights histograms
callbacks.append(tb_callback)

# Early Stopping
# --------------
early_stop = False
if early_stop:
    es_callback = tf.keras.callback.EarlyStopping(monitor='val_loss', patience=10)
    callbacks.append(es_callback)

# ---------------------------------

# How to visualize Tensorboard

# 1. tensorboard --logdir EXPERIMENTS_DIR --port PORT     <- from terminal
# 2. localhost:PORT   <- in your browser


# It is difficult to have a validation metric to evaluate quantitatively the quality
# of a generation, since we are generating new text. Thus, for example, we can 
# evaluate qualitatively the generation performance by generating some text during validation.
# We can do it through a custom callback which, at the beginning of each epoch (on_epoch_begin),
# will take a list of test sequences and use the trained model to generate some text.

class TextGenerationCallback(tf.keras.callbacks.Callback):

  def __init__(self, inference_model, start_sequences, 
               generation_length, temperature):
    super(TextGenerationCallback, self).__init__()

    self.inference_model = inference_model
    self.start_sequences = start_sequences
    self.generation_length = generation_length
    self.temperature = temperature

  def sample(self, pred, temperature=1.0):
    # Helper function to sample an index from a probability array
    # Temperature is a parameter that allows to scale the output of the network, 
    # thus changing the level of 'exploration' in the output distribution. Try
    # yourself to play with this parameter and to see the difference in the generation.
    pred = np.asarray(pred).astype('float64')
    pred = np.log(pred) / temperature
    exp_pred = np.exp(pred)
    pred = exp_pred / np.sum(exp_pred)
    probas = np.random.multinomial(1, pred, 1)
    return np.argmax(probas)

  def generate_text(self, start_sequence, temperature):
    encoded_input = [ctoi[c] for c in start_sequence]
    encoded_input = tf.one_hot(encoded_input, len(vocabulary))
    encoded_input = tf.expand_dims(encoded_input, 0)

    generated_sequence = start_sequence

    in_states = [tf.zeros([1, h_size]) for _ in range(4)]

    for i in range(self.generation_length):
      output = self.inference_model.predict([encoded_input] + in_states)

      in_states = output[1:]
      pred = output[0][0, -1]
      # To get the final prediction we should use the argmax as usual but if we 
      # do so, we will take always the most probable char, so we could incur into a generation loop.
      # Thus, instead of taking the max probability, we sample from the output distribution, 
      # giving a chance to explore also other chars during the generation.
      # We do it with an helper function, that is the 'sample' one. 
      pred = self.sample(pred, temperature)
      pred_char = itoc[pred]
      
      generated_sequence += pred_char

      encoded_input = tf.one_hot(pred, len(vocabulary))
      encoded_input = tf.reshape(encoded_input, [1, 1, len(vocabulary)])

    return generated_sequence

  def on_epoch_begin(self, epoch, logs):
    print("Epoch: {}".format(epoch))
    for start_seq in self.start_sequences:
      print("Starting Sequence: {}".format(start_seq))
      for temp in self.temperature:
        print("Temperature: {}".format(temp))
        generated_seq = self.generate_text(start_seq, temp)
        print(generated_seq)
    return

valid_callback = TextGenerationCallback(inference_model=inference_model,
                                        start_sequences=['Un giorno'], generation_length=100, 
                                        temperature=[0.2, 0.5, 1.0, 1.2])
callbacks.append(valid_callback)

train_model.fit(x=train_dataset,
                epochs=100,  #### set repeat in training dataset
                steps_per_epoch=int(np.ceil(X_enc.shape[0] / bs)),
                callbacks=callbacks)







