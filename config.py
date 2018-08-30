"""
some configurations 
"""

class ResourcesConfig(object):
    data_base_dir = "new_data/"
    model_path = "model/"
    model_name = "single.pkl"

class TrainingConfig(object):
    lr = 1e-3
    batch_size = 512
    epochs = 50
    weight_decay = 1e-5

class ModelConfig(object):
    max_seq_len = 1000
    embedd_size = 50
    vocab_size = 10002
    hidden_size = 64 
    class_num = 19
    n_layers = 3

    top_words = 10000


class Config(object):
    resourses = ResourcesConfig()
    training = TrainingConfig()
    model = ModelConfig()