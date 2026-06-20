import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from continuous_dropouts import *

class MiniBERTWithBottleneck(nn.Module):
    def __init__(self, num_classes, dropout_p, model_name='prajjwal1/bert-mini'):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_config(self.config)

        # Dropout close to the beginning (applied to embeddings)
        self.input_dropout = dropout(dropout_p, 'gaussian')

        # Bottleneck layer for feature representation
        self.bottleneck = nn.Linear(self.config.hidden_size, 512)
        self.relu = nn.ReLU(inplace=True)

        # Final classifier
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, input_ids, attention_mask=None, return_repr=False):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token embedding: outputs.last_hidden_state[:,0,:]
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Apply input dropout to CLS embedding
        x = self.input_dropout(cls_embedding)

        # Bottleneck representation
        rep = self.relu(self.bottleneck(x))

        # Final classification logits
        logits = self.classifier(rep)

        if return_repr:
            return logits, rep
        return logits

    def representation(self, input_ids, attention_mask=None):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token embedding: outputs.last_hidden_state[:,0,:]
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Apply input dropout to CLS embedding
        x = self.input_dropout(cls_embedding)

        # Bottleneck representation
        rep = self.relu(self.bottleneck(x))
        return rep
