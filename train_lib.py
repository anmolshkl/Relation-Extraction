import os
import json

from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
import tensorflow as tf

from data import generate_batches
from util import CLASS_TO_ID


def train(model, optimizer, train_instances, validation_instances, num_epochs, batch_size, serialization_dir):

    print("\nGenerating train batches")
    train_batches = generate_batches(train_instances, batch_size)
    print("\nGenerating val batches")
    val_batches = generate_batches(validation_instances, batch_size)

    train_batch_labels = [batch_inputs.pop("labels") for batch_inputs in train_batches]
    val_batch_labels = [batch_inputs.pop("labels") for batch_inputs in val_batches]
    for epoch in range(num_epochs):
        print(f"\nEpoch{epoch}")

        epoch_loss = 0
        lmbda = 1e-5
        generator_tqdm = tqdm(list(zip(train_batches, train_batch_labels)))
        for batch_inputs, batch_labels in generator_tqdm:
            with tf.GradientTape() as tape:
                logits = model(**batch_inputs, training=True)['logits']
                loss_val = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_labels)
                ### TODO(student) START
                model_params = model.trainable_variables
                regularization = lmbda * tf.add_n([tf.nn.l2_loss(v) for v in model_params])
                ### TODO(Student) END
                loss_val += regularization
                grads = tape.gradient(loss_val, model.trainable_variables)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            loss_val = tf.reduce_mean(loss_val)

            epoch_loss += loss_val

        epoch_loss = epoch_loss / len(train_batches)
        print(f"Train loss for epoch: {epoch_loss}")

        val_loss = 0
        total_preds = []
        total_labels = []
        generator_tqdm = tqdm(list(zip(val_batches, val_batch_labels)))
        for batch_inputs, batch_labels in generator_tqdm:
            logits = model(**batch_inputs, training=False)['logits']
            loss_value = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_labels)
            batch_preds = np.argmax(tf.nn.softmax(logits, axis=-1).numpy(), axis=-1)
            batch_labels = np.argmax(batch_labels, axis=-1)
            total_preds.extend(batch_preds)
            total_labels.extend(batch_labels)
            val_loss += tf.reduce_mean(loss_value)

        # remove "Other" class (id = 0) becase we don't care in evaluation
        non_zero_preds = np.array(list(set(total_preds) - {0}))
        f1 = f1_score(total_labels, total_preds, labels=non_zero_preds, average='macro')
        val_loss = val_loss/len(val_batches)
        print(f"Val loss for epoch: {round(float(val_loss), 4)}")
        print(f"Val F1 score: {round(float(f1), 4)}")

    model.save_weights(os.path.join(serialization_dir, f'model.ckpt'))
    return {'model': model}
