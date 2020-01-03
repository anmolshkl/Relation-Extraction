import os
import json

from tensorflow.keras import models


def load_pretrained_model(serialization_dir: str) -> models.Model:
    """
    Given serialization directory, returns: model loaded with the pretrained weights.
    """

    # Load Config
    config_path = os.path.join(serialization_dir, "config.json")
    model_path = os.path.join(serialization_dir, "model.ckpt.index")

    model_files_present = all([os.path.exists(path)
                               for path in [config_path, model_path]])
    if not model_files_present:
        raise Exception(f"Model files in serialization_dir ({serialization_dir}) "
                        f" are missing. Cannot load_the_model.")

    model_path = model_path.replace(".index", "")

    with open(config_path, "r") as file:
        config = json.load(file)

    # Load Model
    model_name = config.pop("type")
    if model_name == "basic":
        from model import MyBasicAttentiveBiGRU # To prevent circular imports
        model = MyBasicAttentiveBiGRU(**config)
    elif model_name == "advanced":
        from model import MyAdvancedModel # To prevent circular imports
        model = MyAdvancedModel(**config)
    else:
        raise Exception(f"model_name: {model_name} is not supported.")
    model.load_weights(model_path)

    return model


ID_TO_CLASS = {0: 'Other',
               1: 'Message-Topic(e1,e2)', 2: 'Message-Topic(e2,e1)',
               3: 'Product-Producer(e1,e2)', 4: 'Product-Producer(e2,e1)',
               5: 'Instrument-Agency(e1,e2)', 6: 'Instrument-Agency(e2,e1)',
               7: 'Entity-Destination(e1,e2)', 8: 'Entity-Destination(e2,e1)',
               9: 'Cause-Effect(e1,e2)', 10: 'Cause-Effect(e2,e1)',
               11: 'Component-Whole(e1,e2)', 12: 'Component-Whole(e2,e1)',
               13: 'Entity-Origin(e1,e2)', 14: 'Entity-Origin(e2,e1)',
               15: 'Member-Collection(e1,e2)', 16: 'Member-Collection(e2,e1)',
               17: 'Content-Container(e1,e2)', 18: 'Content-Container(e2,e1)'}

CLASS_TO_ID = {_class: index for index, _class in ID_TO_CLASS.items()}

TAG_MAP = [
    ".",
    ",",
    "-LRB-",
    "-RRB-",
    "``",
    "\"\"",
    "''",
    ",",
    "$",
    "#",
    "AFX",
    "CC",
    "CD",
    "DT",
    "EX",
    "FW",
    "HYPH",
    "IN",
    "JJ",
    "JJR",
    "JJS",
    "LS",
    "MD",
    "NIL",
    "NN",
    "NNP",
    "NNPS",
    "NNS",
    "PDT",
    "POS",
    "PRP",
    "PRP$",
    "RB",
    "RBR",
    "RBS",
    "RP",
    "SP",
    "SYM",
    "TO",
    "UH",
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
    "WDT",
    "WP",
    "WP$",
    "WRB",
    "ADD",
    "NFP",
    "GW",
    "XX",
    "BES",
    "HVS",
    "_SP",
]
