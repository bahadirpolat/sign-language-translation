import yaml
from transformers import T5Config, LlamaConfig

class T5ConfigForSLT(T5Config):

    def __init__(
        self,
        sign_input_dim=255,
        sign_vocab_size=512,
        freeze_shared=False,
        sign_embed = 'linear',
        num_conv_layers=3,
        learnable_sign_tokens=False,
        kernel_sizes= [5, 1, 1],
        strides=[1, 1, 1],
        **kwargs
    ):
        super().__init__(**kwargs)

        self.sign_input_dim = sign_input_dim
        self.sign_vocab_size = sign_vocab_size
        self.sign_embed = sign_embed
        self.learnable_sign_tokens = learnable_sign_tokens

        self.num_conv_layers = num_conv_layers
        self.kernel_sizes = kernel_sizes
        self.strides = strides

        assert num_conv_layers == len(kernel_sizes) == len(strides), "Number of convolutional layers, kernel sizes, strides, and output dimensions must be equal."

        self.freeze_shared = freeze_shared

    def load_from_yaml(self, yaml_fpath):
        with open(yaml_fpath, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        for key, value in config.items():
            setattr(self, key, value)
        return self

class LlamaConfigForSLT(LlamaConfig):

    def __init__(
        self,
        multicue=False,
        sign_vocab_size=512,
        pose_vocab_size=512,
        left_hand_vocab_size=512,
        right_hand_vocab_size=512,
        face_vocab_size=512,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.multicue = multicue

        self.sign_vocab_size = sign_vocab_size
        self.pose_vocab_size = pose_vocab_size
        self.left_hand_vocab_size = left_hand_vocab_size
        self.right_hand_vocab_size = right_hand_vocab_size

    @classmethod
    def load_from_yaml(cls, yaml_fpath):
        with open(yaml_fpath, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return cls(**config)
