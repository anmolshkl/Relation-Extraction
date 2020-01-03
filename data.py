# inbuilt lib imports:
from collections import Counter
from typing import List, Dict, Tuple, Any
# external lib imports:
import numpy as np
from tqdm import tqdm
import spacy
import networkx as nx
import util

nlp = spacy.load("en_core_web_sm")


def read_instances(data_file_path: str,
                   max_allowed_num_tokens: int = 150, test: bool=False) -> List[Dict]:
    """
    Reads raw classification dataset from a file and returns a list
    of dicts where each dict defines an instance.
    Parameters
    ----------
    data_file_path : ``str``
        Path to data to be read.
    max_allowed_num_tokens : ``int``
        Maximum number of tokens allowed in the classification instance.
    """
    instances = []
    with open(data_file_path) as file:
        lines = [line for line in file]

    line_inc = 1 if test else 4

    for idx in tqdm(range(0, len(lines), line_inc)):
        instance = dict()

        if not test:
            rel = lines[idx+1].rstrip()
            instance["labels"] = util.CLASS_TO_ID[rel]
        else:
            rel = ''
            instance["labels"] = -1

        sentence_id = lines[idx].split("\t")[0]

        sentence = lines[idx].split("\t")[1][1:-1].lower()

        replacements = [('<e1>', 'e11_'), ('</e1>', '_e12'),
                        ('<e2>', 'e21_'), ('</e2>', '_e22')]

        for replacement in replacements:
            sentence = sentence.replace(replacement[0], replacement[1])

        # clean sentence so entities with hypen/spaces stay as one marker in dep parse
        e11, e12 = sentence.find('e11_'), sentence.find('_e12')
        sentence = sentence[:e11] + sentence[e11:e12+3].replace(' ', '_') + sentence[e12+3:]
        e21, e22 = sentence.find('e21_'), sentence.find('_e22')
        sentence = sentence[:e21] + sentence[e21:e22+3].replace(' ', '_') + sentence[e22+3:]

        e11, e12 = sentence.find('e11_'), sentence.find('_e12')
        sentence = sentence[:e11] + sentence[e11:e12+3].replace('-', '_') + sentence[e12+3:]
        e21, e22 = sentence.find('e21_'), sentence.find('_e22')
        sentence = sentence[:e21] + sentence[e21:e22+3].replace('-', '_') + sentence[e22+3:]

        doc = nlp(sentence)

        e11, e12 = sentence.find('e11_'), sentence.find('_e12')
        e21, e22 = sentence.find('e21_'), sentence.find('_e22')
        start = sentence[e11:e12+4]
        end = sentence[e21:e22+4]

        shortest_path = find_shortest_path(doc, start, end)
        
        # shortest_path = [] 

        if shortest_path:
            tokens = []
            pos = []
            for token in doc:
                if token.lower_ in shortest_path:
                    tokens.append(token.text.lower())
                    pos.append(token.tag_)
        else:
            tokens = [token.text.lower() for token in doc][:max_allowed_num_tokens]
            pos = [token.tag_ for token in doc][:max_allowed_num_tokens]

        instance["text_tokens"] = tokens
        instance["pos_tags"] = pos
        instance["sentence_id"] = sentence_id
        instances.append(instance)

    return instances


def find_shortest_path(doc, start, end):
    edges = []
    for token in doc:
        for child in token.children:
            edges.append(('{0}'.format(token.lower_),
                          '{0}'.format(child.lower_)))

    graph = nx.Graph(edges)

    try:
        shortest_path = nx.shortest_path(graph, start, end)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        shortest_path = []

    return shortest_path

def build_vocabulary(instances: List[Dict],
                     vocab_size: 10000,
                     add_tokens: List[str] = None) -> Tuple[Dict, Dict]:
    """
    Given the instances and max vocab size, this function builds the
    token to index and index to token vocabularies. If list of add_tokens are
    passed, those words will be added first.
    Parameters
    ----------
    instances : ``List[Dict]``
        List of instance returned by read_instances from which we want
        to build the vocabulary.
    vocab_size : ``int``
        Maximum size of vocabulary
    add_tokens : ``List[str]``
        if passed, those words will be added to vocabulary first.
    """
    print("\nBuilding Vocabulary.")

    # make sure pad_token is on index 0
    UNK_TOKEN = "@UNK@"
    PAD_TOKEN = "@PAD@"
    UNK_POS = "@POS@"
    token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1, UNK_POS: 2}

    # First add tokens which were explicitly passed.
    add_tokens = add_tokens or []
    for token in add_tokens:
        if not token.lower() in token_to_id:
            token_to_id[token] = len(token_to_id)

    # Add remaining tokens from the instances as the space permits
    words = []
    for instance in instances:
        words.extend(instance["text_tokens"])
    token_counts = dict(Counter(words).most_common(vocab_size))
    for token, _ in token_counts.items():
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)
        if len(token_to_id) == vocab_size:
            break

    # add pos tags to vocab
    for tag in util.TAG_MAP:
        if tag not in token_to_id:
            token_to_id[tag] = len(token_to_id)
        if len(token_to_id) == vocab_size:
            break

    # Make reverse vocabulary lookup
    id_to_token = dict(zip(token_to_id.values(), token_to_id.keys()))
    return (token_to_id, id_to_token)

def save_vocabulary(vocab_id_to_token: Dict[int, str], vocabulary_path: str) -> None:
    """
    Saves vocabulary to vocabulary_path.
    """
    with open(vocabulary_path, "w") as file:
        # line number is the index of the token
        for idx in range(len(vocab_id_to_token)):
            file.write(vocab_id_to_token[idx] + "\n")

def load_vocabulary(vocabulary_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Loads vocabulary from vocabulary_path.
    """
    vocab_id_to_token = {}
    vocab_token_to_id = {}
    with open(vocabulary_path, "r") as file:
        for index, token in enumerate(file):
            token = token.strip()
            if not token:
                continue
            vocab_id_to_token[index] = token
            vocab_token_to_id[token] = index
    return (vocab_token_to_id, vocab_id_to_token)

def load_glove_embeddings(embeddings_txt_file: str,
                          embedding_dim: int,
                          vocab_id_to_token: Dict[int, str]) -> np.ndarray:
    """
    Given a vocabulary (mapping from index to token), this function builds
    an embedding matrix of vocabulary size in which ith row vector is an
    entry from pretrained embeddings (loaded from embeddings_txt_file).
    """
    tokens_to_keep = set(vocab_id_to_token.values())
    vocab_size = len(vocab_id_to_token)

    embeddings = {}
    print("\nReading pretrained embedding file.")
    with open(embeddings_txt_file) as file:
        for line in tqdm(file):
            line = str(line).strip()
            token = line.split(' ', 1)[0]
            if not token in tokens_to_keep:
                continue
            fields = line.rstrip().split(' ')
            if len(fields) - 1 != embedding_dim:
                raise Exception(f"Pretrained embedding vector and expected "
                                f"embedding_dim do not match for {token}.")
            vector = np.asarray(fields[1:], dtype='float32')
            embeddings[token] = vector

    # Estimate mean and std variation in embeddings and initialize it random normally with it
    all_embeddings = np.asarray(list(embeddings.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_std = float(np.std(all_embeddings))

    embedding_matrix = np.random.normal(embeddings_mean, embeddings_std,
                                        (vocab_size, embedding_dim))
    embedding_matrix = np.asarray(embedding_matrix, dtype='float32')

    for idx, token in vocab_id_to_token.items():
        if token in embeddings:
            embedding_matrix[idx] = embeddings[token]

    return embedding_matrix

def index_instances(instances: List[Dict], token_to_id: Dict) -> List[Dict]:
    """
    Uses the vocabulary to index the fields of the instances. This function
    prepares the instances to be tensorized.
    """
    for instance in instances:
        token_ids = []
        for token in instance["text_tokens"]:
            if token in token_to_id:
                token_ids.append(token_to_id[token])
            else:
                token_ids.append(0) # 0 is index for UNK

        pos_ids = []
        for tag in instance["pos_tags"]:
            if tag in token_to_id:
                pos_ids.append(token_to_id[tag])
            else:
                pos_ids.append(2) # unk for pos

        instance["text_tokens_ids"] = token_ids
        instance["pos_tag_ids"] = pos_ids
        instance.pop("text_tokens")
    return instances


def generate_batches(instances: List[Dict], batch_size) -> List[Dict[str, np.ndarray]]:
    """
    Generates and returns batch of tensorized instances in a chunk of batch_size.
    """

    def chunk(items: List[Any], num: int):
        return [items[index:index+num] for index in range(0, len(items), num)]
    batches_of_instances = chunk(instances, batch_size)

    batches = []
    for batch_of_instances in tqdm(batches_of_instances):

        num_token_ids = [len(instance["text_tokens_ids"])
                         for instance in batch_of_instances]
        max_num_token_ids = max(num_token_ids)

        count = min(batch_size, len(batch_of_instances))
        batch = {"inputs": np.zeros((count, max_num_token_ids), dtype=np.int32),
                 "pos_inputs": np.zeros((count, max_num_token_ids), dtype=np.int32)}

        if "labels" in  batch_of_instances[0]:
            batch["labels"] = np.zeros((count, len(util.CLASS_TO_ID.keys())), dtype=np.int32)

        for batch_index, instance in enumerate(batch_of_instances):
            num_tokens = len(instance["text_tokens_ids"])

            inputs = np.array(instance["text_tokens_ids"])
            batch["inputs"][batch_index][:num_tokens] = inputs

            pos_inputs = np.array(instance["pos_tag_ids"])
            batch["pos_inputs"][batch_index][:num_tokens] = pos_inputs

            if "labels" in instance:
                # Use 1 hot labels
                label = np.zeros(len(util.CLASS_TO_ID.keys()))
                label[instance["labels"]] = 1
                labels = np.array(label)
                batch["labels"][batch_index] = labels

        batches.append(batch)

    return batches
