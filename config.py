"""
some configurations 
"""

class ResourcesConfig(object):
    data_base_dir = "new_data/"
    model_path = "model/"
    model_name = "single.pkl"

class TrainingConfig(object):
    lr = 1e-3
    batch_size = 256
    epochs = 50
    weight_decay = 1e-5

class ModelConfig(object):
    module = None
    MODULES = ["BiLSTM", "BiGRU"]

    max_seq_len = 1024
    embedd_size = 50
    
    hidden_size = 64 
    class_num = 19
    n_layers = 3
    dropout=0.3

    top_words = 10000
    vocab_size = top_words + 2


class Config(object):
    resourses = ResourcesConfig()
    training = TrainingConfig()
    model = ModelConfig()
