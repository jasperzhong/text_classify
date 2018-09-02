"""
some configurations 
"""

class ResourcesConfig(object):
    data_base_dir = "new_data/"
    model_path = "model/"
    model_name = "single.pkl"

class TrainingConfig(object):
    lr = 1e-2
    batch_size = 128
    epochs = 30
    weight_decay = 1e-5
    momentum = 0.9

    fix = True

class ModelConfig(object):
    module = None
    MODULES = ["BiLSTM", "BiGRU"]

    max_seq_len = 1000
    embedd_size = 300
    
    hidden_size = 300 
    class_num = 19
    n_layers = 3
    dropout = 0.3

    d_a = 256
    r = 30

    vocab_size = None


class Config(object):
    mode = None
    resourses = ResourcesConfig()
    training = TrainingConfig()
    model = ModelConfig()
