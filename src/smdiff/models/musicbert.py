import torch
import torch.nn as nn
import math

class MusicBERTConfig:
    def __init__(self, 
                 vocab_sizes=[258, 53, 260, 132, 133, 132, 132, 36], 
                 element_embedding_size=512, 
                 hidden_size=512, 
                 num_layers=4, 
                 num_attention_heads=8, 
                 ffn_inner_hidden_size=2048, 
                 dropout=0.1, 
                 max_position_embeddings=1024,
                 max_seq_len=1024):
        self.vocab_sizes = vocab_sizes
        self.element_embedding_size = element_embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.ffn_inner_hidden_size = ffn_inner_hidden_size
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.max_seq_len = max_seq_len

class MusicBERT(nn.Module):
    def __init__(self, config):
        super(MusicBERT, self).__init__()
        self.config = config
        
        # CHANGE: Use a ModuleList of 8 separate embeddings instead of one shared one
        self.element_embeddings = nn.ModuleList([
            nn.Embedding(config.vocab_sizes[i], config.element_embedding_size) 
            for i in range(8)
        ])
        
        # Linear layer to project concatenated embeddings (8 * 512) to hidden size (512)
        self.linear = nn.Linear(config.element_embedding_size * 8, config.hidden_size)
        
        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, config.max_position_embeddings, config.hidden_size))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size, 
            nhead=config.num_attention_heads, 
            dim_feedforward=config.ffn_inner_hidden_size, 
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # Classifiers for each of the 8 attributes
        self.classifiers = nn.ModuleList([nn.Linear(config.hidden_size, config.vocab_sizes[i]) for i in range(8)])
        
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: (batch_size, seq_len, 8)
        """
        batch_size, seq_length, _ = input_ids.size()
        
        # CHANGE: Embed each attribute separately
        embeds_list = []
        for i in range(8):
            # input_ids[:, :, i] is (batch, seq)
            # embed is (batch, seq, embed_size)
            embed = self.element_embeddings[i](input_ids[:, :, i])
            embeds_list.append(embed)
            
        # Stack to get (batch, seq, 8, embed_size)
        embeds = torch.stack(embeds_list, dim=2)
        
        # 2. Concatenate embeddings
        # (batch_size, seq_len, 8, embed_size) -> (batch_size, seq_len, 8 * embed_size)
        element_embeddings = embeds.view(batch_size, seq_length, -1)
        
        # 3. Linear projection
        x = self.linear(element_embeddings)
        
        # 4. Add Positional Encoding
        # We slice positional encoding to match sequence length
        x = x + self.positional_encoding[:, :seq_length, :]
        x = self.norm(x)
        x = self.dropout(x)
        
        # 5. Transformer Encoder
        # src_key_padding_mask: (batch_size, seq_len)
        x = self.transformer_encoder(x, src_key_padding_mask=attention_mask)
        
        # 6. Classifiers
        # x: (batch_size, seq_len, hidden_size)
        logits = [classifier(x) for classifier in self.classifiers]
        
        # Return list of logits, each (batch_size, seq_len, vocab_size)
        return logits
