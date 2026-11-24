import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    PreTrainedModel,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from .configuration_signt5 import SignT5Config
from transformers.generation.utils import GenerateOutput

from dataclasses import dataclass
from typing import List, Optional, Union, Callable, Tuple

@dataclass
class PreTrainingOutputForSignT5:

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    diversity_loss: Optional[torch.FloatTensor] = None

class SignConv1D(nn.Module):
    """
    1D Convolutional layer with GELU activation function.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(SignConv1D, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.gelu = ACT2FN["gelu"] if "gelu" in ACT2FN else F.gelu

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.conv1d.weight)
        nn.init.zeros_(self.conv1d.bias)

    def forward(self, x):
        y = self.conv1d(x)
        y = self.gelu(y)
        return y

class SignGumbelQuantizer(nn.Module):
    """
    Gumbel quantizer for sign embeddings.
    """

    def __init__(self, temperature=1.0):
        super(SignGumbelQuantizer, self).__init__()
        self.temperature = temperature

    def _set_gumbel_temperature(self, temperature):
        self.temperature = temperature
        
    def forward(self, x):
        noise = torch.rand_like(x)
        noise = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
        y = F.gumbel_softmax(x + noise, tau=self.temperature, hard=True)
        return y

AUTO_MODEL_FOR_SEQ2SEQ_LM_MAPPING = {
    "google/mt5-small": "google/mt5-small",
    "google/mt5-base": "google/mt5-base",
    "google/mt5-large": "google/mt5-large",
    "boun-tabi-LMG/TURNA": "boun-tabi-LMG/TURNA",
}

AUTO_MODEL_FOR_CAUSAL_LM_MAPPING = {
    "ytu-ce-cosmos/turkish-gpt2": "ytu-ce-cosmos/turkish-gpt2",
}

class SignT5(PreTrainedModel):
    config_class = SignT5Config

    def __init__(self, config: SignT5Config):
        super().__init__(config)

        self.config = config
        if config.base_model_name in AUTO_MODEL_FOR_SEQ2SEQ_LM_MAPPING:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name)
            self.base_model_dim = self.model.config.d_model
        elif config.base_model_name in AUTO_MODEL_FOR_CAUSAL_LM_MAPPING:
            self.model = AutoModelForCausalLM.from_pretrained(config.base_model_name)
            self.base_model_dim = self.model.config.n_embd
        
        if config.base_model_type == 't5':
            if config.sign_embed == "mlp": # This will be change to MLP
                self.sign_embedding = nn.Sequential(
                    nn.Linear(config.sign_input_dim, self.base_model_dim),
                    nn.Dropout(config.hidden_dropout_prob),
                    nn.GELU(),
                )
            elif config.sign_embed == "linear":
                self.sign_embedding = nn.Sequential(
                    nn.Linear(config.sign_input_dim, self.base_model_dim),
                    nn.Dropout(config.hidden_dropout_prob),
                )
            elif config.sign_embed == "conv":
                # Add gumbel softmax layer
                self.sign_embedding = nn.Sequential(
                    *[
                        # First add Conv1d layers to reduce the dimensionality
                        SignConv1D(
                            in_channels=(
                                config.sign_input_dim
                                if layer_id == 0
                                else self.base_model_dim
                            ),
                            out_channels=self.base_model_dim,
                            kernel_size=config.kernel_sizes[layer_id],
                            stride=config.strides[layer_id],
                        )
                        for layer_id in range(config.num_conv_layers)
                    ]
                )
                self.sign_conv_linear = nn.Sequential(
                    nn.Linear(self.base_model_dim, self.base_model_dim),
                    nn.Dropout(config.hidden_dropout_prob),
                    nn.GELU(),
                )
            else:
                raise ValueError("Invalid sign embedding type.")

        self._init_sign_weights()

        if config.base_model_type == 'signvq':
            self.sign_embedding = self.base_model.sign_embedding

        self.model.generation_config = GenerationConfig(
            # Use base model config for beam search
            top_k=self.model.generation_config.top_k,
            top_p=self.model.generation_config.top_p,
            max_length=self.model.generation_config.max_length,
            do_sample=self.model.generation_config.do_sample,
            temperature=self.model.generation_config.temperature,
            num_beams=self.model.generation_config.num_beams,
            no_repeat_ngram_size=self.model.generation_config.no_repeat_ngram_size,
            length_penalty=self.model.generation_config.length_penalty,
            repetition_penalty=self.model.generation_config.repetition_penalty,
            early_stopping=self.model.generation_config.early_stopping,
            decoder_start_token_id=self.model.config.pad_token_id,
            eos_token_id=self.model.config.eos_token_id,
        )

    def freeze_shared(self):
        for param in self.model.shared.parameters():
            param.requires_grad = False
    ###################################################################### I added alignment tokens for alg task *alg
    """
    def add_alignment_tokens(self, tokenizer):
        Add special tokens for alignment task
        special_tokens = ["<slt>", "<alignment>"]
    
        # Add timestamp tokens for reasonable range (0-7200 seconds = 2 hours)
        for i in range(7200):
            special_tokens.extend([f"<start_{i}>", f"<end_{i}>"])
    
        # Add tokens to tokenizer
        num_added = tokenizer.add_tokens(special_tokens)
    
        # Resize model embeddings to accommodate new tokens
        if num_added > 0:
            self.model.resize_token_embeddings(len(tokenizer))
    
        return num_added
    """
    def add_alignment_tokens(self, tokenizer):
        """
        Add task tokens to the tokenizer and resize model embeddings.
        - '<slt>' token for standard Sign Language Translation task
        - '<alignment>' token for temporal alignment task
        
        Timestamps like "0 --> 7" are learned as regular text.
        """
        # Only 2 task tokens needed
        special_tokens = ['<slt>', '<alignment>']
        
        # Add tokens to tokenizer
        num_added = tokenizer.add_tokens(special_tokens)
        
        if num_added > 0:
            # Resize model embeddings to accommodate new tokens
            self.model.resize_token_embeddings(len(tokenizer))
            print(f"Added {num_added} special tokens: {special_tokens}")
            print(f"New vocabulary size: {len(tokenizer)}")
        else:
            print(f"Tokens {special_tokens} already exist.")
            
        return num_added

    ######################################################################

    def _init_sign_weights(self):
        if self.config.sign_embed == "linear" or self.config.sign_embed == "mlp":
            for layer in self.sign_embedding:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)
        elif self.config.sign_embed == "conv":
            for layer in self.sign_embedding:
                if isinstance(layer, SignConv1D):
                    nn.init.xavier_uniform_(layer.conv1d.weight)
                    nn.init.constant_(layer.conv1d.bias, 0.0)
            for layer in self.sign_conv_linear:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)
            
    def _calculate_sign_output_shape(self, input_length):
        """
        Calculate the output shape of the sign embedding layer.
        This is required to calculate the input shape of the positional encoding layer.
        """
        input_shape = (1, input_length)
        for layer_id in range(self.config.num_conv_layers):
            input_shape = (
                input_shape[0],
                int(
                    (input_shape[1] - self.config.kernel_sizes[layer_id])
                    / self.config.strides[layer_id]
                )
                + 1,
            )
        return input_shape[1]

    def _get_sign_attention_mask(self, sign_attention_mask):
        """
        Generate attention mask for sign inputs.
        """
        if self.config.sign_embed == "conv":
            masks = [
                torch.ones(
                    1,
                    math.ceil(
                        self._calculate_sign_output_shape(
                            sign_attention_mask[idx][
                                sign_attention_mask[idx] != 0
                            ].size(0)
                        )
                    ) + 2,
                    dtype=torch.int,
                ).squeeze(0)
                for idx in range(sign_attention_mask.shape[0])
            ]
            sign_attention_mask = pad_sequence(masks, batch_first=True, padding_value=0)
        return sign_attention_mask

    def _embed_sign(self, sign_inputs):
        if self.config.sign_embed == "conv":
            sign_inputs = sign_inputs.transpose(1, 2)
            sign_embeds = self.sign_embedding(sign_inputs)
            sign_inputs = sign_embeds.transpose(1, 2)
            # Apply linear layer to sign embeddings
            sign_inputs = self.sign_conv_linear(sign_inputs)
            return sign_inputs

        if self.config.sign_embed == "linear" or self.config.sign_embed == "mlp":
            return self.sign_embedding(sign_inputs)

    def _embed_text_sign(
        self,
        text_inputs: List[torch.LongTensor] = None,
        sign_inputs: torch.LongTensor = None,
        text_attention_mask: Optional[torch.FloatTensor] = None,
        sign_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        
        # Embed text inputs
        text_embeds = self.model.shared(text_inputs)
        
        text_attention_mask  = text_attention_mask.to('cpu')

        inputs_embeds, attention_mask = [], []

        for idx in range(text_embeds.size(0)):
            non_zero_text_indices = text_attention_mask[idx][
                text_attention_mask[idx] != 0
            ]
            non_zero_sign_indices = sign_attention_mask[idx][
                sign_attention_mask[idx] != 0
            ]

            # Concatenate text and sign embeddings
            inputs_embeds.append(
                torch.cat(
                    [
                        sign_inputs[idx][non_zero_sign_indices],
                        text_embeds[idx][non_zero_text_indices],
                    ]
                )
            )

            # Concatenate text and sign attention masks
            attention_mask.append(
                torch.cat(
                    [
                        sign_attention_mask[idx][non_zero_sign_indices],
                        text_attention_mask[idx][non_zero_text_indices],
                    ]
                )
            )

        inputs_embeds = pad_sequence(inputs_embeds, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        # Convert size(2) of inputs_embeds to pad_token_embedding 
        pad_token_embedding = self.model.shared.weight.data[self.model.config.pad_token_id]
        inputs_embeds = torch.where(inputs_embeds == 0, pad_token_embedding, inputs_embeds)

        return inputs_embeds, attention_mask
    
    def _get_frame_to_position_mapping(self, text_inputs, sign_inputs, text_attention_mask, sign_attention_mask):
        """
        Find and return the mapping of frames to positions in the input embeddings.
        """
        if self.config.sign_embed == "linear" or self.config.sign_embed == "ln":
            # Get the non-zero indices of the sign inputs
            sign_indices = [
                sign_attention_mask[idx][sign_attention_mask[idx] != 0]
                for idx in range(sign_attention_mask.size(0))
            ]
            # Get the non-zero indices of the text inputs
            text_indices = [
                text_attention_mask[idx][text_attention_mask[idx] != 0]
                for idx in range(text_attention_mask.size(0))
            ]
            # Concatenate the sign and text indices
            indices = [torch.cat([sign_indices[idx], text_indices[idx]]) for idx in range(len(sign_indices))]
            # Get the frame to position mapping
            frame_to_position_mapping = [
                torch.argsort(indices[idx], descending=False) for idx in range(len(indices))
            ]
            return frame_to_position_mapping
        
        if self.config.sign_embed == "conv":
            # Get the non-zero indices of the sign inputs
            sign_indices = [
                sign_attention_mask[idx][sign_attention_mask[idx] != 0]
                for idx in range(sign_attention_mask.size(0))
            ]
            # Get the non-zero indices of the text inputs
            text_indices = [
                text_attention_mask[idx][text_attention_mask[idx] != 0]
                for idx in range(text_attention_mask.size(0))
            ]

            raise NotImplementedError("Frame to position mapping for conv sign embeddings is not implemented yet.")
    
    def _get_attention_map_between_text_and_sign(
            self, 
            text_inputs, 
            sign_inputs, 
            text_attention_mask, 
            sign_attention_mask,
            encoder_layer_idx=0,
        ):
        """
        Get the attention map between text and sign inputs.
        """

        # Get the frame to position mapping
        frame_to_position_mapping = self._get_frame_to_position_mapping(
            text_inputs, sign_inputs, text_attention_mask, sign_attention_mask
        )   

        # Get the attention map between text and sign inputs
        attention_map = self.model.encoder.get_attention_map(
            text_inputs=text_inputs,
            sign_inputs=sign_inputs,
            text_attention_mask=text_attention_mask,
            sign_attention_mask=sign_attention_mask,
            frame_to_position_mapping=frame_to_position_mapping,
            encoder_layer_idx=encoder_layer_idx,
        )

    def _embed_input_sequences(
        self,
        text_inputs: List[torch.LongTensor] = None,
        sign_inputs: torch.LongTensor = None,
        text_attention_mask: Optional[torch.FloatTensor] = None,
        sign_attention_mask: Optional[torch.FloatTensor] = None,
    ):
        """
        Embed input sequences for the model. This method is used to embed input sequences for the model.

        Args:
            text_inputs (:obj:`List[torch.LongTensor]`):
                List of input sequences in the form of torch.LongTensor.
            sign_inputs (:obj:`torch.LongTensor`):
                Input sequences in the form of torch.LongTensor.
            text_attention_mask (:obj:`Optional[torch.FloatTensor]`, `optional`):
                Attention mask for text inputs.
            sign_attention_mask (:obj:`Optional[torch.FloatTensor]`, `optional`):
                Attention mask for sign inputs.

        Returns:
            :obj:`Tuple[torch.FloatTensor, torch.FloatTensor]`: Tuple of input embeddings and attention mask.

        """

        if sign_inputs is None:
            raise ValueError("Sign inputs must be provided.")
        
        # print("Sign inputs shape:", sign_inputs.shape)
        # print("Sign attention mask shape:", sign_attention_mask.shape)
        
        # Remove zero padding from sign inputs until vector is in self.sign_input_dim
        sign_inputs = sign_inputs[:, :, : self.config.sign_input_dim]
        # Embed sign inputs
        inputs_embeds = self._embed_sign(sign_inputs)
        attention_mask = self._get_sign_attention_mask(sign_attention_mask)

        # Pad attention mask if necessary
        if inputs_embeds.size(1) < attention_mask.size(1):
            attention_mask = attention_mask[:, : inputs_embeds.size(1)]

        # print("Sign embeddings shape:", inputs_embeds.shape)
        # print("Sign attention mask shape:", attention_mask.shape)

        # if text_inputs is not None and self.config.embed_text:
        #     inputs_embeds, attention_mask = self._embed_text_sign(
        #         text_inputs=text_inputs,
        #         sign_inputs=inputs_embeds,
        #         text_attention_mask=text_attention_mask,
        #         sign_attention_mask=attention_mask,
        #     )

        # to device
        inputs_embeds = inputs_embeds.to(self.model.shared.weight.device)
        attention_mask = attention_mask.to(self.model.shared.weight.device)

        return inputs_embeds, attention_mask

    def forward(
        self,
        text_inputs: List[torch.LongTensor] = None,
        sign_inputs: torch.LongTensor = None,
        text_attention_mask: Optional[torch.FloatTensor] = None,
        sign_attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        """
        Forward pass for the model.

        Args:
            text_inputs (:obj:`List[torch.LongTensor]`):
                List of input sequences in the form of torch.LongTensor.
            sign_inputs (:obj:`torch.LongTensor`):
                Input sequences in the form of torch.LongTensor.
            text_attention_mask (:obj:`Optional[torch.FloatTensor]`, `optional`):
                Attention mask for text inputs.
            sign_attention_mask (:obj:`Optional[torch.FloatTensor]`, `optional`):
                Attention mask for sign inputs.

        Returns:
            :obj:`Tuple[torch.FloatTensor, torch.FloatTensor]`: Tuple of input embeddings and attention mask.

        """

        if inputs_embeds is None:
            inputs_embeds, attention_mask = self._embed_input_sequences(
                text_inputs=text_inputs,
                sign_inputs=sign_inputs,
                text_attention_mask=text_attention_mask,
                sign_attention_mask=sign_attention_mask,
            )
    
        if self.config.learnable_sign_tokens:
            # Quantize sign embeddings
            sign_input_tokens = self.quantizer(inputs_embeds)
            print("Sign input tokens:", sign_input_tokens)
            # Get sign input embeddings
            inputs_embeds = self.sign_input_embedding(sign_input_tokens.argmax(dim=-1))

            prob = sign_input_tokens.sum(dim=(0, 1)) / sign_input_tokens.sum()
            diversity_loss = -torch.sum(prob * torch.log(prob + 1e-8))

        outputs = self.model(
            input_ids=None,  # We don't need input_ids since we provide inputs_embeds
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.learnable_sign_tokens:   
            
            loss = outputs.loss
            loss += self.config.diversity_loss_weight * diversity_loss

            outputs['loss'] = outputs.loss
            outputs['diversity_loss'] = diversity_loss
    
        return outputs

    @torch.no_grad()
    def generate(
        self,
        text_inputs: List[torch.LongTensor] = None,
        sign_inputs: torch.LongTensor = None,
        text_attention_mask: Optional[torch.FloatTensor] = None,
        sign_attention_mask: Optional[torch.FloatTensor] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        synced_gpus: Optional[bool] = None,
        streamer=None,
        assistant_model: Optional["PreTrainedModel"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Generate output sequences from the model.

        Args:
            text_inputs (:obj:`List[torch.LongTensor]`):
                List of input sequences in the form of torch.LongTensor.
            sign_inputs (:obj:`torch.LongTensor`):
                Input sequences in the form of torch.LongTensor.
            text_attention_mask (:obj:`Optional[torch.FloatTensor]`, `optional`):
                Attention mask for text inputs.
            sign_attention_mask (:obj:`Optional[torch.FloatTensor]`, `optional`):
                Attention mask for sign inputs.

        Returns:
            :obj:`torch.LongTensor`: Generated output sequences.

        """

        if input_embeds is None:
            inputs_embeds, attention_mask = self._embed_input_sequences(
                text_inputs=text_inputs,
                sign_inputs=sign_inputs,
                text_attention_mask=text_attention_mask,
                sign_attention_mask=sign_attention_mask,
            )

        return self.model.generate(
            input_ids=None,  # We don't need input_ids since we provide inputs_embeds
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            streamer=streamer,
            assistant_model=assistant_model,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs,
        )

class SignT5ForPretraining(SignT5):

    def __init__(self, config: SignT5Config, mode='train'):
        super().__init__(config)

        assert config.embed_text == False, "SignT5ForPretraining does not support text embeddings."

        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name)
        self.sign_tokenizer = SignGumbelQuantizer(temperature=config.temperature)
        self.sign_embedding = nn.Embedding(config.sign_vocab_size, self.base_model_dim)

        if self.config.learnable_sign_tokens:
            # But we need to discritize the sign embeddings
            self.quantizer = SignGumbelQuantizer(temperature=config.temperature)
            self.sign_input_embedding = nn.Embedding(config.sign_vocab_size, self.base_model_dim)
            # Do Random initialization
            nn.init.xavier_uniform_(self.sign_input_embedding.weight)
            # We implement random projection for sign embeddings
            # Thats why need to freeze the sign input embeddings
            self.sign_input_embedding.requires_grad_(False)

        self.mode = mode

    def _set_gumbel_temperature(self, temperature):
        self.sign_tokenizer._set_gumbel_temperature(temperature)

    def forward(
        self,
        sign_inputs: torch.LongTensor = None,
        sign_attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        mask_indices: torch.LongTensor = None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):

        if inputs_embeds is None:
            inputs_embeds, attention_mask = self.SignT5._embed_input_sequences(
                text_inputs=None,
                sign_inputs=sign_inputs,
                text_attention_mask=None,
                sign_attention_mask=sign_attention_mask,
            )

        # Quantize sign embeddings
        sign_input_tokens = self.sign_tokenizer(inputs_embeds)

        # Create t5 pretraining 
        # 1. Embed sign tokens
        # 2. Get random mask indices for sign tokens such as 15% of the tokens
        # Such as "1, 24, 2566, 325, 2677" 
        #  Input ids:  1, 24, <MASK>, 2677
        #  Label ids:  2566, 325 <MASK>

        input_ids = sign_input_tokens
        mask_indices = torch.randperm(input_ids.size(1))[: int(input_ids.size(1) * 0.15)]
        labels = input_ids.clone()
        labels[:, mask_indices] = -100

        # 2. Get label embeddings
        decoder_inputs_embeds = self.sign_embedding(labels)

        # 3. Calculate diversity loss for indices
        usage_counts = sign_input_tokens.sum(dim=(0, 1))
        probabilities = usage_counts / usage_counts.sum()
        diversity_loss = -torch.sum(probabilities * torch.log(probabilities + 1e-8))  # Add epsilon to avoid log(0)

        # 4. Calculate the loss
        model_output = self.SignT5.model(
            input_ids=None,  # We don't need input_ids since we provide inputs_embeds
            inputs_embeds=signs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        return PreTrainingOutputForSignT5(
            loss=model_output.loss,
            logits=model_output.logits,
            hidden_states=model_output.hidden_states,
            attentions=model_output.attentions,
            diversity_loss=diversity_loss,
        )
    

