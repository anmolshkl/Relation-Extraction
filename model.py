import tensorflow as tf
from tensorflow.keras import layers, models

from util import ID_TO_CLASS
import pdb

class MyBasicAttentiveBiGRU(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyBasicAttentiveBiGRU, self).__init__()

        self.num_classes = len(ID_TO_CLASS)

        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas = tf.Variable(tf.random.normal((hidden_size*2, 1)))

        ### TODO(Students) START
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)), trainable = training) 
        
        self._forward_layer = layers.GRU(hidden_size, return_sequences=True)
        self._backward_layer = layers.GRU(hidden_size, return_sequences=True, go_backwards=True)
        self._bidirectional_layer = layers.Bidirectional(self._forward_layer, backward_layer=self._backward_layer, merge_mode = 'concat')
        ### TODO(Students) END

    def attn(self, rnn_outputs):
        ### TODO(Students) START

        # The following lines are based on the equation 9-12 by Zhou et al.
        
        # Equation 9: M = tanh(H)
        M = tf.tanh(rnn_outputs) # 10, 5, 256

        # Equation 10: \alpha = softmax(w^T.M)
        alpha = tf.nn.softmax(tf.tensordot(M, self.omegas, axes=1))  # 10, 5, 1

        # Equation 11: H.\alpha^T
        r = tf.reduce_sum(rnn_outputs * alpha, axis=1)

        # Equation 12: tanh(r)
        output = tf.tanh(r)

        ### TODO(Students) END

        return output

    def call(self, inputs, pos_inputs, training):
        
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs) # 10, 5, 100
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs) # 10, 5, 100
        
        ### TODO(Students) START

        # First, we will concatenate the word and POS embeddings along the 3rd axis
        input_for_bidirectional = tf.concat([word_embed, pos_embed], axis=2) # 10, 5, 200
        
        # input_for_bidirectional = word_embed # This is for experiments, comment the above line

        # Create a mask for the sequence as it is padded with 0
        mask = tf.cast(tf.greater(inputs, 0), tf.float32)

        # Pass the word (+ POS) embeddings to bidirectional GRU layer
        hidden_outputs = self._bidirectional_layer(input_for_bidirectional, mask = mask) # 10, 5, 256

        # Apply attention to the hidden state outputs from forward and backward layer of biGRU
        attn = self.attn(hidden_outputs)

        # Apply the dense layer to obtain logits
        logits = self.decoder(attn)
        ### TODO(Students) END

        # batch_size, num_classes
        return {'logits': logits}


class MyAdvancedModel(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyAdvancedModel, self).__init__()

        ### TODO(Students) START
        self.num_classes = len(ID_TO_CLASS)

        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)), trainable = training)

        # I tried masking but Conv1D does not support masking as of TF 2.0 and 
        # therefore adding a masking layer did not make any difference - https://github.com/keras-team/keras/issues/411
        # Adding or removing this does not have any effect on F1 score/loss
        self._masking = layers.Masking(mask_value=0, input_shape=(None, 200))

        # Conv1D with filter size of 256 and kernel size of 3
        self.conv1 = layers.Conv1D(filters=256, kernel_size=3, 
                padding="valid", activation="tanh", 
                strides=1)
        # Using GlobalMaxPool1D as MaxPool1D does not support variable sequence length
        self.pool1 = layers.GlobalMaxPool1D()

        # Conv1D with filter size of 128 and kernel size of 3
        self.conv2 = layers.Conv1D(filters=128, kernel_size=3, 
                        padding="valid", activation="tanh", 
                        strides=1)
        self.pool2 = layers.GlobalMaxPool1D()

        # Conv1D with filter size of 64 and kernel size of 3
        self.conv3 = layers.Conv1D(filters=64, kernel_size=3, 
                        padding="valid", activation="tanh",
                        strides=1)
        self.pool3 = layers.GlobalMaxPool1D()

        # Layer to concatenate the output of all the CNN layers
        self.concatenate = layers.Concatenate()

        # Dropout for inputs
        self.dropout1 = layers.Dropout(0.5)
        # Dropout for second last dense layer
        self.dropout2 = layers.Dropout(0.5)

        self.dense1 = layers.Dense(100, activation="tanh")
        self.dense2 = layers.Dense(self.num_classes)

        ### TODO(Students END

    def call(self, inputs, pos_inputs, training):
        ### TODO(Students) START
        # pdb.set_trace()
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs) # 10, 5, 100
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs) # 10, 5, 100
        
        concatenated_input = tf.concat([word_embed, pos_embed], axis=2) # 10, 5, 200

        masked_input = self._masking(concatenated_input)
        dropped_input = self.dropout1(masked_input)

        # Convolution layer 1
        conv1_output = self.conv1(dropped_input)
        conv1_output = self.pool1(conv1_output)

        # Convolution layer 2
        conv2_output = self.conv2(dropped_input)
        conv2_output = self.pool2(conv2_output)

        # Convolution layer 3
        conv3_output = self.conv3(dropped_input)
        conv3_output = self.pool3(conv3_output)

        # Concatenate the output of all convolution layers
        conv_outputs = self.concatenate([conv1_output, conv2_output, conv3_output])
        conv_outputs = self.dropout2(conv_outputs)

        conv_outputs = self.dense1(conv_outputs)
        logits = self.dense2(conv_outputs)

        return {'logits': logits}
        ### TODO(Students END