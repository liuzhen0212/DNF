import torch
# from torchsearchsorted import searchsorted
from torch import searchsorted

__all__ = ['render_rays']


# single model
def render_rays_lensless(models, embeddings, rays):
    ## rays,        [batch, kernel_h, kernel_w, 2]
    model_coarse = models[0]
    embedding_xy = embeddings[0]
    xy_embedding = embedding_xy(rays)
    results = model_coarse(xy_embedding)
    return results

# double models
def render_rays_lensless_2stream(models, embeddings, rays):
    ## rays,        [batch, kernel_h, kernel_w, 2] -> [-1, 2]
    model_amp = models[0]
    model_phase = models[1]
    embedding_amp = embeddings[0]
    embedding_phase = embeddings[1]
    amp_embedding = embedding_amp(rays)
    phase_embedding = embedding_phase(rays)
    results_amp = model_amp(amp_embedding)
    results_phase = model_phase(phase_embedding)
    return results_amp, results_phase