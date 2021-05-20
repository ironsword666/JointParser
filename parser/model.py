# -*- coding: utf-8 -*-

from parser.modules import CHAR_LSTM, MLP, BertEmbedding, Biaffine, BiLSTM, TreeCRFLoss
from parser.modules.dropout import IndependentDropout, SharedDropout

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args
        self.pretrained = False
        # the embedding layer
        self.char_embed = nn.Embedding(num_embeddings=args.n_chars,
                                       embedding_dim=args.n_embed)
        n_lstm_input = args.n_embed
        if args.feat == 'bert':
            self.feat_embed = BertEmbedding(model=args.bert_model,
                                            n_layers=args.n_bert_layers,
                                            n_out=args.n_feat_embed)
            n_lstm_input += args.n_feat_embed
        if self.args.feat in {'bigram', 'trigram'}:
            self.bigram_embed = nn.Embedding(num_embeddings=args.n_bigrams,
                                             embedding_dim=args.n_embed)
            n_lstm_input += args.n_embed
        if self.args.feat == 'trigram':
            self.trigram_embed = nn.Embedding(num_embeddings=args.n_trigrams,
                                              embedding_dim=args.n_embed)
            n_lstm_input += args.n_embed

        self.embed_dropout = IndependentDropout(p=args.embed_dropout)

        # the lstm layer
        self.lstm = BiLSTM(input_size=n_lstm_input,
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
                                  n_out=args.n_sublabels,
                                  bias_x=True,
                                  bias_y=True)

        self.label_attn = Biaffine(n_in=args.n_mlp_label,
                                   n_out=args.n_labels,
                                   bias_x=True,
                                   bias_y=True)

        self.crf = TreeCRFLoss(n_labels=args.n_sublabels) 
        
        self.pad_index = args.pad_index
        self.unk_index = args.unk_index

    def load_pretrained(self, embed_dict=None):
        embed = embed_dict['embed'] if isinstance(
            embed_dict, dict) and 'embed' in embed_dict else None
        if embed is not None:
            self.pretrained = True
            self.char_pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.char_embed.weight)
            if self.args.feat == 'bigram':
                embed = embed_dict['bi_embed']
                self.bi_pretrained = nn.Embedding.from_pretrained(embed)
                nn.init.zeros_(self.bigram_embed.weight)
            elif self.args.feat == 'trigram':
                bi_embed = embed_dict['bi_embed']
                tri_embed = embed_dict['tri_embed']
                self.bi_pretrained = nn.Embedding.from_pretrained(bi_embed)
                self.tri_pretrained = nn.Embedding.from_pretrained(tri_embed)
                nn.init.zeros_(self.bigram_embed.weight)
                nn.init.zeros_(self.trigram_embed.weight)
        return self

    def forward(self, feed_dict):
        chars = feed_dict["chars"]
        batch_size, seq_len = chars.shape
        # get the mask and lengths of given batch
        mask = chars.ne(self.pad_index)
        lens = mask.sum(dim=1)
        ext_chars = chars
        # set the indices larger than num_embeddings to unk_index
        if self.pretrained:
            ext_mask = chars.ge(self.char_embed.num_embeddings)
            ext_chars = chars.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        char_embed = self.char_embed(ext_chars)
        if self.pretrained:
            char_embed += self.char_pretrained(chars)

        if self.args.feat == 'bert':
            feats = feed_dict["feats"]
            feat_embed = self.feat_embed(*feats)
            char_embed, feat_embed = self.embed_dropout(char_embed, feat_embed)
            embed = torch.cat((char_embed, feat_embed), dim=-1)
        elif self.args.feat == 'bigram':
            bigram = feed_dict["bigram"]
            ext_bigram = bigram
            if self.pretrained:
                ext_mask = bigram.ge(self.bigram_embed.num_embeddings)
                ext_bigram = bigram.masked_fill(ext_mask, self.unk_index)
            bigram_embed = self.bigram_embed(ext_bigram)
            if self.pretrained:
                bigram_embed += self.bi_pretrained(bigram)
            char_embed, bigram_embed = self.embed_dropout(
                char_embed, bigram_embed)
            embed = torch.cat((char_embed, bigram_embed), dim=-1)
        elif self.args.feat == 'trigram':
            bigram = feed_dict["bigram"]
            trigram = feed_dict["trigram"]
            ext_bigram = bigram
            ext_trigram = trigram
            if self.pretrained:
                ext_mask = bigram.ge(self.bigram_embed.num_embeddings)
                ext_bigram = bigram.masked_fill(ext_mask, self.unk_index)
                ext_mask = trigram.ge(self.trigram_embed.num_embeddings)
                ext_trigram = trigram.masked_fill(ext_mask, self.unk_index)
            bigram_embed = self.bigram_embed(ext_bigram)
            trigram_embed = self.trigram_embed(ext_trigram)
            if self.pretrained:
                bigram_embed += self.bi_pretrained(bigram)
                trigram_embed += self.tri_pretrained(trigram)
            char_embed, bigram_embed, trigram_embed = self.embed_dropout(
                char_embed, bigram_embed, trigram_embed)
            embed = torch.cat(
                (char_embed, bigram_embed, trigram_embed), dim=-1)
        else:
            embed = self.embed_dropout(char_embed)[0]

        x = pack_padded_sequence(embed, lens, True, False)
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

        # [batch_size, seq_len, seq_len, n_sublabels]
        s_span = self.span_attn(span_l, span_r).permute(0, 2, 3, 1)
        # [batch_size, seq_len, seq_len, n_labels]
        s_label = self.label_attn(label_l, label_r).permute(0, 2, 3, 1)

        return s_span, s_label, self.crf.transitions, self.crf.start_transitions

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
        if self.pretrained:
            pretrained = {'embed': state_dict.pop('char_pretrained.weight')}
            if hasattr(self, 'bi_pretrained'):
                pretrained.update(
                    {'bi_embed': state_dict.pop('bi_pretrained.weight')})
            if hasattr(self, 'tri_pretrained'):
                pretrained.update(
                    {'tri_embed': state_dict.pop('tri_pretrained.weight')})
        state = {
            'args': self.args,
            'state_dict': state_dict,
            'pretrained': pretrained
        }
        torch.save(state, path)
