import torch
from ecg_jepa_feature_extracting_v3 import ecg_jepa

def load_encoder(ckpt_dir, leads=None):

    if leads is None:
        leads = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

    params = {
        'encoder_embed_dim': 768,
        'encoder_depth': 12,
        'encoder_num_heads': 16,
        'predictor_embed_dim': 384,
        'predictor_depth': 6,
        'predictor_num_heads': 12,
        'c': 19,
        'pos_type': 'sincos',
        'mask_scale': (0, 0),
        'leads': leads
    }
    encoder = ecg_jepa(**params).encoder
    ckpt = torch.load(ckpt_dir)
    encoder.load_state_dict(ckpt['encoder'])
    embed_dim = 768

    return encoder, embed_dim
