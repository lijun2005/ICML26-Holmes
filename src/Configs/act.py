import os
import yaml

cfg = {}


cfg['model_name'] = 'Holmes'
cfg['dataset_name'] = 'activitynet'
cfg['seed'] = 0
cfg['root']='/data2/lianniu/'
cfg['data_root']='/data2/lianniu/DreamPRVR/'
cfg['visual_feature'] = 'i3d'
cfg['collection'] = 'activitynet'
cfg['map_size'] = 32
cfg['clip_scale_w'] = 0.6
cfg['frame_scale_w'] = 0.4

cfg['model_root'] = ''
cfg['ckpt_path'] = ''


# extra
cfg['sft_factor'] = 0.6


# dataset
cfg['num_workers'] = 32
cfg['no_core_driver'] = False
cfg['no_pin_memory'] = False
cfg['batchsize'] = 128
cfg['visual_feat_dim'] = 1024  


# opt
cfg['lr'] = 0.00025
cfg['lr_warmup_proportion'] = 0.01
cfg['wd'] = 0.01
cfg['margin'] = 0.2


# train
cfg['n_epoch'] = 100
cfg['max_es_cnt'] = 10
cfg['hard_negative_start_epoch'] = 20
cfg['hard_pool_size'] = 20
cfg['use_hard_negative'] = False
cfg['loss_factor'] = [0.02, 0.04, 0.003, 0.0001,0.06,0.]
cfg['neg_factor'] = [0.2, 32, 1]


# eval
cfg['eval_query_bsz'] = 50
cfg['eval_context_bsz'] = 100


# model
cfg['max_desc_l'] = 30
cfg['max_ctx_l'] = 128
cfg['sub_feat_size'] = 768
cfg['q_feat_size'] = 1024
cfg['max_position_embeddings'] = 300
cfg['hidden_size'] = 384
cfg['n_heads'] = 4
cfg['input_drop'] = 0.31
cfg['drop'] = 0.31
cfg['initializer_range'] = 0.02


cfg['num_workers'] = 1 if cfg['no_core_driver'] else cfg['num_workers']
cfg['pin_memory'] = not cfg['no_pin_memory']



cfg['pot_scale'] = 0.1 
cfg['pot_prompt_ratio'] = 0.3


# cfg['edl_tau'] = 0.15


# cfg['edl_intra_video_tau'] = 0.08
cfg['edl_tau'] = 0.04

cfg['edl_intra_video_tau'] = 0.08

cfg['warmup_epoch'] = 20
cfg['warmup_end'] = False



cfg['division_beta'] = 0.3
cfg['smooth_beta'] = 0.2


def get_cfg_defaults():
    return cfg