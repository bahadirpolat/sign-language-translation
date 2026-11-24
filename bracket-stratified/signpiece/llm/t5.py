import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from transformers.modeling_outputs import (
    ModelOutput,
)

import warnings
from signpiece.utils.beam_search import BeamSearch
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

@dataclass
class Seq2SeqVQLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    vq_sequence_output: Optional[torch.FloatTensor] = None

class SignConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(SignConv1d, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride
        )
        self.gelu = nn.GELU()

        self._init_weights()

    def _init_weights(self):
        nn.init.normal(self.conv1d.weight)
    
    def forward(self, x):
        y = self.conv1d(x)
        y = self.gelu(y)
        return y

class SignGumbelSoftmax(nn.Module):

    def __init__(self, temperature, hard=False):
        super(SignGumbelSoftmax, self).__init__()
        self.temperature = temperature
        self.hard = hard

    def forward(self, x):
        y = F.gumbel_softmax(x, tau=self.temperature, hard=self.hard)
        return y

class SignT5(nn.Module):
    def __init__(self, model_name_or_path, config, **kwargs):
        super(SignT5, self).__init__()

        self.config = config
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.device = self.base_model.shared.weight.device

        if config.beam_search:
            self.beam_search = BeamSearch(
                max_length=config.max_length,
                max_new_tokens=config.max_new_tokens,
                length_penalty=config.length_penalty,
                top_k=config.top_k,
                top_p=config.top_p,
                temperature=config.temperature,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample
            )
            self.beam_search.eos_token_id = config.eos_token_id

        if config.sign_embed == 'conv':
            # Add gumbel softmax layer
            self.sign_embedding = nn.Sequential(*[
                # First add Conv1d layers to reduce the dimensionality
                SignConv1d(
                    in_channels=config.sign_input_dim if layer_id == 0 else config.d_model,
                    out_channels=config.d_model,
                    kernel_size=config.kernel_sizes[layer_id],
                    stride=config.strides[layer_id]
                )
                for layer_id in range(config.num_conv_layers)
            ])
        
        if config.learnable_sign_tokens:
            self.gumbel_softmax = SignGumbelSoftmax(
                temperature=config.gumbel_temperature, hard=True
            )
            self.sign_codebook = nn.Embedding(config.sign_vocab_size, config.d_model)

        if config.sign_embed == 'linear':
            self.sign_embedding = nn.Sequential(
                nn.Linear(config.sign_input_dim, config.d_model),
                nn.GELU()
            )

        self.init_sign_weights()

    def init_sign_weights(self):
        if self.config.sign_embed == 'linear':
            for layer in self.sign_embedding:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)
        elif self.config.sign_embed == 'conv':
            for layer in self.sign_embedding:
                if isinstance(layer, SignConv1d):
                    nn.init.xavier_uniform_(layer.conv1d.weight)
                    nn.init.constant_(layer.conv1d.bias, 0.0)

    def set_gumbel_temperature(self, temperature):
        self.gumbel_softmax.temperature = temperature

    def freeze_shared(self):
        for param in self.base_model.shared.parameters():
            param.requires_grad = False

    def embed_sign(self, sign_inputs):
        if self.config.sign_embed == 'conv':
            sign_inputs = sign_inputs.transpose(1,2)
            sign_embeds = self.sign_embedding(sign_inputs)
            sign_inputs = sign_embeds.transpose(1,2)
            return sign_inputs
        
        if self.config.sign_embed == 'linear':
            return self.sign_embedding(sign_inputs)

    def _calculate_sign_output_shape(self, input_length):
        input_shape = (1, input_length)
        for layer_id in range(self.config.num_conv_layers):
            input_shape = (
                input_shape[0],
                int((input_shape[1] - self.config.kernel_sizes[layer_id]) / self.config.strides[layer_id]) + 1
            )
        return input_shape[1]
    
    def generate(
        self, **model_kwargs
    ):
        inputs_embeds, attention_mask = self._embed_input_sequences(
            text_inputs=model_kwargs.get("text_inputs"),
            sign_inputs=model_kwargs.get("sign_inputs"),
            text_attention_mask=model_kwargs.get("text_attention_mask"),
            sign_attention_mask=model_kwargs.get("sign_attention_mask")
        )

        model_kwargs.pop("text_inputs", None)
        model_kwargs.pop("sign_inputs", None)
        model_kwargs.pop("text_attention_mask", None)
        model_kwargs.pop("sign_attention_mask", None)

        if self.beam_search is None:
            return self.base_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                num_beams=self.base_model.generation_config.num_beams,
                top_k=self.base_model.generation_config.top_k,
                top_p=self.base_model.generation_config.top_p,
                temperature=self.base_model.generation_config.temperature,
                max_length=self.base_model.generation_config.max_length,
                min_length=self.base_model.generation_config.min_length,
                early_stopping=self.base_model.generation_config.early_stopping,
                length_penalty=self.base_model.generation_config.length_penalty,
                no_repeat_ngram_size=self.base_model.generation_config.no_repeat_ngram_size,
                eos_token_id=self.config.eos_token_id,
            )
        else:
            return self.beam_search.generate(
                model=self,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **model_kwargs
            )

    def _embed_input_sequences(
            self,
            text_inputs : List[torch.LongTensor] = None,
            sign_inputs : torch.LongTensor = None,
            text_attention_mask : Optional[torch.FloatTensor] = None,
            sign_attention_mask : Optional[torch.FloatTensor] = None,
            only_sign : bool = False
        ):
        """
        Embed input sequences for the model. This method is used to embed input sequences for the model.

        # This is done for variable context length such as 
            # Example 1: 
            # - Text: 
            #    Translate ASL to English with context:
            #    Context: "I went to the store and bought some apples"
            # - Sign:
            #     [FRAME_1, FRAME_2, ..., FRAME_N]
            # Example 2:
            # - Text:
            #    Translate ASL to English with context:
            #    Context: "I went to the store and bought some apples and oranges"
            # - Sign:
            #     [FRAME_1, FRAME_2, ..., FRAME_M]

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

        # Check if text and sign inputs has any nan values
        if torch.isnan(text_inputs).any():
            print("NaN values detected in text inputs")
        if torch.isnan(sign_inputs).any():
            print("NaN values detected in sign inputs")

        # Check if text and sign inputs has any nan values
        if torch.isnan(text_inputs).any():
            print("NaN values detected in text inputs")
        if torch.isnan(sign_inputs).any():
            print("NaN values detected in sign inputs")

        # Embed text inputs
        text_embeds = self.base_model.shared(text_inputs)
        # Remove zero padding from sign inputs until vector is in self.sign_input_dim
        sign_inputs = sign_inputs[:, :, :self.config.sign_input_dim]
        # Embed sign inputs
        sign_embeds = self.embed_sign(sign_inputs)
        
        # Check if sign inputs has any nan values
        if torch.isnan(sign_embeds).any():
            print("NaN values detected in sign embeddings")

        if torch.isnan(text_embeds).any():
            print("NaN values detected in text embeddings")

        # If conv layers are used, reduce the dimensionality of the sign attention mask
        if self.config.sign_embed == 'conv':
            # print("Sign Attention Mask Shape: ", sign_attention_mask)
            masks = [
                torch.ones(
                    1,
                    math.ceil(self._calculate_sign_output_shape(
                        sign_attention_mask[idx][sign_attention_mask[idx] != 0].size(0)
                    )) + 2, dtype=torch.int).squeeze(0) 
                for idx in range(sign_attention_mask.shape[0])
            ]
            sign_attention_mask = pad_sequence(masks, batch_first=True, padding_value=0)
    
        # No Mombo Jumbo here
        inputs_embeds = sign_embeds
        attention_mask = sign_attention_mask

        # to device
        inputs_embeds = inputs_embeds.to(self.base_model.shared.weight.device)
        attention_mask = attention_mask.to(self.base_model.shared.weight.device)
        
        return inputs_embeds, attention_mask

        if only_sign:

            # Concatenate text and sign embeddings
            # Add </s> token to the end of the input
            # Add pad token embedding until max length
            # Format [text_shape[0] + sign_shape[0] + 1 + (<EOS> token)]
            inputs_embeds = torch.cat([
                text_embeds, 
                sign_embeds, 
            ], dim=1)

            sign_attention_mask = sign_attention_mask.to(inputs_embeds.device)
            # Create attention mask by padding with 0
            # Format [text_shape[0] + sign_shape[0] + 1 + (<s> token)]
            attention_mask = torch.cat([
                text_attention_mask, 
                sign_attention_mask, 
            ],dim=1)

            # to device
            inputs_embeds = inputs_embeds.to(self.base_model.shared.weight.device)
            attention_mask = attention_mask.to(self.base_model.shared.weight.device)
            
            return inputs_embeds, attention_mask

            # Create NEW INPUTS EMBEDS with only sign embeddings by filtering out padding from sign_attention_mask
            inputs_embeds = torch.stack([
                torch.cat(
                    [
                        # Add sign embeddings
                        sign_embeds[idx][sign_attention_mask[idx][sign_attention_mask[idx] != 0]],
                        # Add </s> token to the end of the input
                        self.model.shared.weight[self.config.eos_token_id].unsqueeze(0),
                        # Add pad token embedding until max length
                        self.model.shared.weight[self.config.pad_token_id].unsqueeze(0).repeat(
                            max(
                                sign_attention_mask.shape[1] + 2 - 
                                sign_attention_mask[idx][sign_attention_mask[idx] != 0].shape[0],
                                1
                            ), 1
                        )
                    ]
                )
                for idx in range(sign_embeds.shape[0])
            ])
            # Create attention mask by padding with 0 for [sign_shape[1] + 1 + (<s> token)]
            attention_mask = pad_sequence([
                torch.ones(
                    # Create attention mask by padding with 0 for [sign_shape[1] + (<s> token) + (<pad> token) +  ... ]
                    sign_attention_mask[idx][sign_attention_mask[idx] != 0].shape[0] + 2, dtype=torch.int).squeeze(0)
                for idx in range(sign_embeds.shape[0])
            ], batch_first=True, padding_value=0)

            # Add additional padding to attention mask to match the inputs_embeds shape
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.zeros(
                        attention_mask.shape[0],
                        inputs_embeds.shape[1] - attention_mask.shape[1],
                    )
                ]
            , dim=1)

        else:
            # Check if text and sign inputs has any nan values
            if torch.isnan(text_embeds).any() or torch.isnan(sign_embeds).any():
                print("NaN values detected in text or sign embeddings")
            
            # Create attention mask by padding with 0
            # Format [text_shape[0] + sign_shape[0] + 1 + (<s> token)] 
            attention_mask = pad_sequence([
                torch.ones((
                    # Text and sign embeddings without padding
                    text_embeds[idx][text_attention_mask[idx][text_attention_mask[idx] != 0]].shape[0] +
                    sign_embeds[idx][sign_attention_mask[idx][sign_attention_mask[idx] != 0]].shape[0] +
                    2, # <EOS> token AND padding
                ))
                for idx in range(text_embeds.shape[0])
            ], batch_first=True, padding_value=0)     

            # Check if attention mask has any nan values
            if torch.isnan(attention_mask).any():
                print("NaN values detected in attention mask")
            
            # Create new tensor with text and sign embeddings concatenated
            inputs_embeds = torch.stack([
                torch.cat(
                    [
                        # Add text and sign embeddings
                        text_embeds[idx][text_attention_mask[idx][text_attention_mask[idx] != 0]],
                        sign_embeds[idx][sign_attention_mask[idx][sign_attention_mask[idx] != 0]],
                        # Add </s> token to the end of the input
                        self.base_model.shared.weight[self.config.eos_token_id].unsqueeze(0),
                        # Add pad token embedding until max length
                        self.base_model.shared.weight[self.config.pad_token_id].unsqueeze(0).repeat(
                            max(
                                attention_mask.shape[1] - 
                                text_attention_mask[idx][text_attention_mask[idx] != 0].shape[0] - 
                                sign_attention_mask[idx][sign_attention_mask[idx] != 0].shape[0] - 1,
                                1
                            ), 1
                        )
                    ]
                )
                for idx in range(text_embeds.shape[0])
            ])

            # Check if inputs_embeds has any nan values
            if torch.isnan(inputs_embeds).any():
                print("NaN values detected in inputs embeddings")

        # replace nan values with 0
        inputs_embeds[inputs_embeds != inputs_embeds] = 0
        
        inputs_embeds = inputs_embeds.to(self.base_model.shared.weight.device)
        attention_mask = attention_mask.to(self.base_model.shared.weight.device)

        return inputs_embeds, attention_mask

    # Override the forward method to modify input embeddings
    def forward(
        self,
        sign_inputs=None,
        text_inputs=None,
        text_attention_mask=None,
        sign_attention_mask=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        inputs_embeds=None,
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
            if text_inputs is not None and sign_inputs is not None:
                inputs_embeds, attention_mask = self._embed_input_sequences(
                    text_inputs=text_inputs,
                    sign_inputs=sign_inputs,
                    text_attention_mask=text_attention_mask,
                    sign_attention_mask=sign_attention_mask
                )
            else:
                raise ValueError("You have to specify either inputs or inputs_embeds")


        # Pass modified embeddings to the original T5 forward method
        return self.base_model.forward(
            input_ids=None,  # We use inputs_embeds instead of input_ids
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            inputs_embeds=inputs_embeds,
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