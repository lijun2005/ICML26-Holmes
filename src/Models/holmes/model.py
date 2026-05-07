import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
from Models.holmes.model_components import BertAttention, LinearLayer, \
                                            TrainablePositionalEncoding, DyGMMBlock
from Losses.partialot import POT,BatchPOT
import ipdb


class Holmes_Net(nn.Module):
    def __init__(self, config):
        super(Holmes_Net, self).__init__()
        self.config = config

        self.query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l,
                                                           hidden_size=config.hidden_size, dropout=config.input_drop)
        self.clip_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                         hidden_size=config.hidden_size, dropout=config.input_drop)
        self.frame_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                          hidden_size=config.hidden_size, dropout=config.input_drop)
        #
        self.query_input_proj = LinearLayer(config.query_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.query_encoder = BertAttention(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop))


        self.clip_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.clip_encoder = DyGMMBlock(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop, frame_len=32, sft_factor=config.sft_factor,attention_num=8))

        self.frame_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                             dropout=config.input_drop, relu=True)
        self.frame_encoder_1 = DyGMMBlock(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop, frame_len=128, sft_factor=config.sft_factor,attention_num=8))
                    
        self.modular_vector_mapping = nn.Linear(config.hidden_size, out_features=1, bias=False)

        self.weight_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))

        self.pot = POT(config)
        self.batchpot = BatchPOT(config)
        self.reset_parameters()

    def reset_parameters(self):
        """ Initialize the weights."""

        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(re_init)

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        """use_hard_negative: bool; hard_pool_size: int, """
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size


    def forward(self, batch):

        clip_video_feat = batch['clip_video_features']
        query_feat = batch['text_feat']
        query_mask = batch['text_mask']
        query_labels = batch['text_labels']

        frame_video_feat = batch['frame_video_features']
        frame_video_mask = batch['videos_mask']

        encoded_frame_feat, vid_proposal_feat = self.encode_context(
            clip_video_feat, frame_video_feat, frame_video_mask)

        clip_scale_scores, clip_scale_scores_, frame_scale_scores, frame_scale_scores_ \
            = self.get_pred_from_raw_query(
            query_feat, query_mask, query_labels, vid_proposal_feat, encoded_frame_feat, return_query_feats=True)

        label_dict = {}
        for index, label in enumerate(query_labels):
            if label in label_dict:
                label_dict[label].append(index)
            else:
                label_dict[label] = []
                label_dict[label].append(index)

        video_query = self.encode_query(query_feat, query_mask)
        
 
        video_indices = list(label_dict.keys())
        clip_embs_list = [vid_proposal_feat[k] for k in video_indices]
        text_embs_list = [video_query[label_dict[k]] for k in video_indices]

        max_queries = max(t.shape[0] for t in text_embs_list)
        
        padded_text_embs_list = []
        text_mask_list = []
        
        for t in text_embs_list:
            num_queries = t.shape[0]
            

            mask = torch.ones(num_queries, device=t.device)
            
            if num_queries < max_queries:

                padding = torch.zeros((max_queries - num_queries, t.shape[1]), device=t.device, dtype=t.dtype)
                padded_t = torch.cat([t, padding], dim=0)

                mask_padding = torch.zeros(max_queries - num_queries, device=t.device)
                mask = torch.cat([mask, mask_padding], dim=0)
            else:
                padded_t = t

            padded_text_embs_list.append(padded_t)
            text_mask_list.append(mask)


        batched_clip_embs = torch.stack(clip_embs_list, dim=0) 
        batched_text_embs = torch.stack(padded_text_embs_list, dim=0)
        batched_text_mask = torch.stack(text_mask_list, dim=0)


        batched_intravideo_sim,batch_pot_results, pot_mass = self.batchpot(batched_clip_embs, batched_text_embs, batched_text_mask)




        return [clip_scale_scores, clip_scale_scores_, label_dict, frame_scale_scores, frame_scale_scores_, video_query, 
                batched_intravideo_sim, batch_pot_results, batched_text_mask]


    def encode_query(self, query_feat, query_mask):
        encoded_query = self.encode_input(query_feat, query_mask, self.query_input_proj, self.query_encoder,
                                          self.query_pos_embed)  # (N, Lq, D)
        if query_mask is not None:
            mask = query_mask.unsqueeze(1)
        
        video_query = self.get_modularized_queries(encoded_query, query_mask)  # (N, D) * 1

        return video_query


    def encode_context(self, clip_video_feat, frame_video_feat, video_mask=None):

        encoded_clip_feat = self.encode_input(clip_video_feat, None, self.clip_input_proj, self.clip_encoder,
                                               self.clip_pos_embed, self.weight_token)

        if frame_video_feat.shape[1] != 128:
            fix = 128 - frame_video_feat.shape[1]
            temp_feat = 0.0 * frame_video_feat.mean(dim=1, keepdim=True).repeat(1, fix, 1)
            frame_video_feat = torch.cat([frame_video_feat, temp_feat], dim=1)

            temp_mask = 0.0 * video_mask.mean(dim=1, keepdim=True).repeat(1, fix).type_as(video_mask)
            video_mask = torch.cat([video_mask, temp_mask], dim=1)
        
        encoded_frame_feat = self.encode_input(frame_video_feat, video_mask, self.frame_input_proj,
                                                self.frame_encoder_1,
                                                self.frame_pos_embed, self.weight_token)

        encoded_frame_feat = torch.where(video_mask.unsqueeze(-1).repeat(1, 1, encoded_frame_feat.shape[-1]) == 1.0, \
                                                                        encoded_frame_feat, 0. * encoded_frame_feat)

        return encoded_frame_feat, encoded_clip_feat

    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer, weight_token=None):
        """
        Args:
            feat: (N, L, D_input), torch.float32
            mask: (N, L), torch.float32, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            pos_embed_layer: positional embedding layer
        """

        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor
        if weight_token is not None:
            return encoder_layer(feat, mask, weight_token)  # (N, L, D_hidden)
        else:
            return encoder_layer(feat, mask)  # (N, L, D_hidden)


    def get_modularized_queries(self, encoded_query, query_mask):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        modular_attention_scores = self.modular_vector_mapping(encoded_query)  # (N, L, 2 or 1)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 2 or 1, D)
        return modular_queries.squeeze()


    @staticmethod
    def get_clip_scale_scores(modularied_query, context_feat):

        modularied_query = F.normalize(modularied_query, dim=-1)
        context_feat = F.normalize(context_feat, dim=-1)

        clip_level_query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)

        query_context_scores, indices = torch.max(clip_level_query_context_scores,
                                                  dim=1)  # (N, N) diagonal positions are positive pairs
        
        return query_context_scores


    @staticmethod
    def get_unnormalized_clip_scale_scores(modularied_query, context_feat):

        query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)

        output_query_context_scores, indices = torch.max(query_context_scores, dim=1)

        return output_query_context_scores


    def get_pred_from_raw_query(self, query_feat, query_mask, query_labels=None,
                                video_proposal_feat=None, encoded_frame_feat=None,
                                return_query_feats=False):

        video_query = self.encode_query(query_feat, query_mask)

        # get clip-level retrieval scores
        clip_scale_scores = self.get_clip_scale_scores(
            video_query, video_proposal_feat)

        frame_scale_scores = self.get_clip_scale_scores(
            video_query, encoded_frame_feat)

        if return_query_feats:
            clip_scale_scores_ = self.get_unnormalized_clip_scale_scores(video_query, video_proposal_feat)
            frame_scale_scores_ = self.get_unnormalized_clip_scale_scores(video_query, encoded_frame_feat)

            return clip_scale_scores, clip_scale_scores_, frame_scale_scores, frame_scale_scores_
        else:

            return clip_scale_scores, frame_scale_scores


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)
