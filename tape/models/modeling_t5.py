# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modified by Roshan Rao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch T5 model. """

import logging

import torch
from torch import nn
from transformers.modeling_t5 import T5Config as HFT5Config
from tape.models.modeling_utils import LayerNorm

from ..registry import registry
from .modeling_utils import ProteinConfig, ProteinModel, MLMHead
from .modeling_utils import T5Stack

logger = logging.getLogger(__name__)

URL_PREFIX = "https://storage.googleapis.com/fire-tod.tryps.in/pytorch-models/"
T5_PRETRAINED_MODEL_ARCHIVE_MAP = {}
T5_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class ProteinT5Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be
        # able to load any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class T5Config(ProteinConfig):
    pretrained_config_archive_map = T5_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size: int = 30,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 dropout_prob: float = 0.1,
                 max_position_embeddings: int = 8096,
                 layer_norm_eps: float = 1e-12,
                 gradient_checkpointing: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps

        # Adapt comparable argument names from BertConfig for consistency
        self.dropout_rate = dropout_prob
        self.d_ff = intermediate_size
        self.d_model = hidden_size
        self.num_layers = num_hidden_layers
        self.num_heads = num_attention_heads
        self.n_positions = max_position_embeddings
        self.use_cache = False
        self.checkpoint = gradient_checkpointing


class T5AbstractModel(ProteinModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = T5Config
    pretrained_model_archive_map = T5_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "t5"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


@registry.register_task_model('embed', 't5enc')
class T5Model(T5AbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.embedding = ProteinT5Embeddings(config)

        self.model = T5Stack(config)

        self.init_weights()

    def forward(self,
                input_ids,
                input_mask=None):
        return self.model(inputs_embeds=self.embedding(input_ids),
                          attention_mask=input_mask)


@registry.register_task_model('masked_language_modeling', 't5enc')
class T5ForMLM(MLMHead, T5AbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.model = T5Model(config)

        self.mlm = MLMHead(
            config.hidden_size, config.vocab_size, config.hidden_act, config.layer_norm_eps,
            ignore_index=-1)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.mlm.decoder,
                                   self.model.embeddings.word_embeddings)

    def forward(self,
                input_ids,
                input_mask=None,
                targets=None):

        outputs = self.model(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        # add hidden states and attention if they are here
        outputs = self.mlm(sequence_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs
