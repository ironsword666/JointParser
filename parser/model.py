# -*- coding: utf-8 -*-

from parser.modules import CHAR_LSTM, MLP, BertEmbedding, Biaffine, BiLSTM
from parser.modules.dropout import IndependentDropout, SharedDropout

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args
        # the embedding layer
        n_feat_embed = 0
        self.word_embed = nn.Embedding(num_embeddings=args.n_words,
                                       embedding_dim=args.n_embed)
        if args.feat == 'char':
            self.feat_embed = CHAR_LSTM(n_chars=args.n_feats,
                                        n_embed=args.n_char_embed,
                                        n_out=args.n_feat_embed)
            n_feat_embed = args.n_feat_embed
        elif args.feat == 'bert':
            self.feat_embed = BertEmbedding(model=args.bert_model,
                                            n_layers=args.n_bert_layers,
                                            n_out=args.n_feat_embed)
            n_feat_embed = args.n_feat_embed

        self.embed_dropout = IndependentDropout(p=args.embed_dropout)

        # the lstm layer
        self.lstm = BiLSTM(input_size=args.n_embed+n_feat_embed,
                           hidden_size=args.n_lstm_hidden,
                           num_layers=args.n_lstm_layers,
                           dropout=args.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=args.lstm_dropout)

        # the MLP layers
        self.mlp_span_l = MLP(n_in=args.n_lstm_hidden*2,
                              n_out=args.n_mlp_span,
                              dropout=args.mlp_dropout)
        self.mlp_span_r = MLP(n_in=args.n_lstm_hidden*2,
                              n_out=args.n_mlp_span,
                              dropout=args.mlp_dropout)
        self.mlp_label_l = MLP(n_in=args.n_lstm_hidden*2,
                               n_out=args.n_mlp_label,
                               dropout=args.mlp_dropout)
        self.mlp_label_r = MLP(n_in=args.n_lstm_hidden*2,
                               n_out=args.n_mlp_label,
                               dropout=args.mlp_dropout)

        # the Biaffine layers
        self.span_attn = Biaffine(n_in=args.n_mlp_span,
                                  bias_x=True,
                                  bias_y=False)
        self.label_attn = Biaffine(n_in=args.n_mlp_label,
                                   n_out=args.n_labels,
                                   bias_x=True,
                                   bias_y=True)
        self.pad_index = args.pad_index
        self.unk_index = args.unk_index

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)

        return self

    def forward(self, words, feats):
        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        lens = mask.sum(dim=1)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)
        if self.args.feat == 'char':
            feat_embed = self.feat_embed(feats[mask])
            feat_embed = pad_sequence(feat_embed.split(lens.tolist()), True)
            word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
            # concatenate the word and feat representations
            embed = torch.cat((word_embed, feat_embed), dim=-1)
        elif self.args.feat == 'bert':
            feat_embed = self.feat_embed(*feats)
            word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
            # concatenate the word and feat representations
            embed = torch.cat((word_embed, feat_embed), dim=-1)
        else:
            embed = self.embed_dropout(word_embed)[0]

        x = pack_padded_sequence(embed, lens.cpu(), True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)

        x_f, x_b = x.chunk(2, dim=-1)
        x = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)
        # apply MLPs to the BiLSTM output states
        span_l = self.mlp_span_l(x)
        span_r = self.mlp_span_r(x)
        label_l = self.mlp_label_l(x)
        label_r = self.mlp_label_r(x)

        # [batch_size, seq_len, seq_len]
        s_span = self.span_attn(span_l, span_r)
        # [batch_size, seq_len, seq_len, n_labels]
        s_label = self.label_attn(label_l, label_r).permute(0, 2, 3, 1)

        return s_span, s_label

    @classmethod
    def load(cls, path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(path, map_location=device)
        model = cls(state['args'])
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(device)

        return model

    def save(self, path):
        state_dict, pretrained = self.state_dict(), None
        if hasattr(self, 'pretrained'):
            pretrained = state_dict.pop('pretrained.weight')
        state = {
            'args': self.args,
            'state_dict': state_dict,
            'pretrained': pretrained
        }
        torch.save(state, path)
