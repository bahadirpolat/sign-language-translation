from transformers import PretrainedConfig
from typing import List


class SignT5Config(PretrainedConfig):

    model_type = "signt5"

    def __init__(
        self,
        base_model_name: str = "t5-small",
        sign_input_dim=255,
        sign_vocab_size=512,
        freeze_shared=False,
        sign_embed = 'linear',
        hidden_dropout_prob=0.1,
        num_conv_layers=3,
        kernel_sizes= [5, 1, 1],
        strides=[1, 1, 1],
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        beam_search_type: str = "default", # default to custom beam search
        learnable_sign_tokens=True,
        diversity_loss_weight: float = 1.0,
        base_model_type: str = "t5",
        **kwargs
    ) -> None:
        """
        Initializes SignT5 configuration.
        
        Args:
            base_model_name (str): The base model name of the T5 model.
            sign_input_dim (int): The dimension of the sign input.
            sign_vocab_size (int): The size of the sign vocabulary.
            freeze_shared (bool): Whether to freeze the shared weights of the model.
            sign_embed (str): The type of sign embedding to use.
            hidden_dropout_prob (float): The dropout probability for the hidden layers.
            num_conv_layers (int): The number of convolutional layers.
            learnable_sign_tokens (bool): Whether to make the sign tokens learnable.
            kernel_sizes (List[int]): The kernel sizes for the convolutional layers.
            strides (List[int]): The strides for the convolutional layers.
            pad_token_id (int): The token ID for padding.
            bos_token_id (int): The token ID for the beginning of sentence.
            eos_token_id (int): The token ID for the end of sentence.
            beam_search_type (str): The type of beam search to use.
                
        Returns:
            None
        """
        super().__init__(**kwargs)

        self.base_model_name = base_model_name
        self.sign_input_dim = sign_input_dim
        self.sign_vocab_size = sign_vocab_size
        self.hidden_dropout_prob = hidden_dropout_prob

        self.sign_embed = sign_embed
        self.num_conv_layers = num_conv_layers
        self.kernel_sizes = kernel_sizes
        self.strides = strides

        assert num_conv_layers == len(kernel_sizes) == len(strides), "Number of convolutional layers, kernel sizes, strides, and output dimensions must be equal."

        self.freeze_shared = freeze_shared

        self.beam_search_type = beam_search_type

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.learnable_sign_tokens = learnable_sign_tokens
        self.base_model_type = base_model_type


class SignT5ForPretrainingConfig(PretrainedConfig):

    model_type = "signt5"

    def __init__(
        self,
        base_model_name: str = "t5-small",
        sign_input_dim=255,
        sign_vocab_size=512,
        freeze_shared=False,
        sign_embed = 'linear',
        hidden_dropout_prob=0.1,
        num_conv_layers=3,
        kernel_sizes= [5, 1, 1],
        strides=[1, 1, 1],
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        beam_search_type: str = "default", # default to custom beam search
        learnable_sign_tokens=True,
        diversity_loss_weight: float = 1.0,
        base_model_type: str = "t5",
        **kwargs
    ) -> None:
        """
        Initializes SignT5 configuration.
        
        Args:
            base_model_name (str): The base model name of the T5 model.
            sign_input_dim (int): The dimension of the sign input.
            sign_vocab_size (int): The size of the sign vocabulary.
            freeze_shared (bool): Whether to freeze the shared weights of the model.
            sign_embed (str): The type of sign embedding to use.
            hidden_dropout_prob (float): The dropout probability for the hidden layers.
            num_conv_layers (int): The number of convolutional layers.
            learnable_sign_tokens (bool): Whether to make the sign tokens learnable.
            kernel_sizes (List[int]): The kernel sizes for the convolutional layers.
            strides (List[int]): The strides for the convolutional layers.
            pad_token_id (int): The token ID for padding.
            bos_token_id (int): The token ID for the beginning of sentence.
            eos_token_id (int): The token ID for the end of sentence.
            beam_search_type (str): The type of beam search to use.
                
        Returns:
            None
        """
        super().__init__(**kwargs)

        self.base_model_name = base_model_name
        self.sign_input_dim = sign_input_dim
        self.sign_vocab_size = sign_vocab_size
        self.hidden_dropout_prob = hidden_dropout_prob

        self.sign_embed = sign_embed
        self.num_conv_layers = num_conv_layers
        self.kernel_sizes = kernel_sizes
        self.strides = strides

        assert num_conv_layers == len(kernel_sizes) == len(strides), "Number of convolutional layers, kernel sizes, strides, and output dimensions must be equal."

        self.freeze_shared = freeze_shared

        self.beam_search_type = beam_search_type

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.learnable_sign_tokens = learnable_sign_tokens
        self.base_model_type = base_model_type