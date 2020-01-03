import argparse
import os
import json
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers

from train_lib import train
from data import read_instances, build_vocabulary, \
    save_vocabulary, index_instances, load_glove_embeddings
from model import MyAdvancedModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--data-file', type=str, help="Location of data", default="./data/train.txt")
    parser.add_argument('--val-file', type=str, help="Location of val data", default="./data/val.txt")
    parser.add_argument('--batch-size', type=int, help="size of batch", default=10)
    parser.add_argument('--epochs', type=int, help="num epochs", default=10)
    parser.add_argument('--embed-file', type=str, help="embedding location", default='./data/glove.6B.100D.txt')
    parser.add_argument('--embed-dim', type=int, help="size of embeddings", default=100)

    args = parser.parse_args()

    tf.random.set_seed(1337)
    np.random.seed(1337)
    random.seed(13370)

    MAX_TOKENS = 250
    VOCAB_SIZE = 10000
    GLOVE_COMMON_WORDS_PATH = os.path.join("data", "glove_common_words.txt")

    print(f"\nReading Train Instances")
    train_instances = read_instances(args.data_file, MAX_TOKENS)
    print(f"\nReading Val Instances")
    val_instances = read_instances(args.val_file, MAX_TOKENS)

    with open(GLOVE_COMMON_WORDS_PATH) as file:
        glove_common_words = [line.strip() for line in file.readlines() if line.strip()]

    vocab_token_to_id, vocab_id_to_token = build_vocabulary(train_instances, VOCAB_SIZE,
                                                            glove_common_words)
    vocab_size = len(np.unique(vocab_token_to_id.keys())[0])

    train_instances = index_instances(train_instances, vocab_token_to_id)
    val_instances = index_instances(val_instances, vocab_token_to_id)

    ### TODO(Students) START
    # make a config file here as expected by your MyAdvancedModel
    config = {'vocab_size': vocab_size, 'embed_dim': args.embed_dim, 'training': True}
    ### TODO(Students END
    model = MyAdvancedModel(**config)
    config['type'] = 'advanced'

    optimizer = optimizers.Adam()

    embeddings = load_glove_embeddings(args.embed_file, args.embed_dim, vocab_id_to_token)
    model.embeddings.assign(tf.convert_to_tensor(embeddings))

    save_serialization_dir = os.path.join('serialization_dirs', 'advanced')
    if not os.path.exists(save_serialization_dir):
        os.makedirs(save_serialization_dir)

    train_output = train(model, optimizer, train_instances, val_instances,
                         args.epochs, args.batch_size, save_serialization_dir)

    config_path = os.path.join(save_serialization_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f)

    vocab_path = os.path.join(save_serialization_dir, "vocab.txt")
    save_vocabulary(vocab_id_to_token, vocab_path)

    print(f"\nModel stored in directory: {save_serialization_dir}")
