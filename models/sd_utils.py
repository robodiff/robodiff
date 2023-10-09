#%%
import torch
import numpy as np
import math

def sdBox(p, b):
    d = torch.abs(p) - b

    external_dist = torch.norm(torch.max(d, torch.zeros_like(d)), dim=0)
    
    d_maxed = torch.max(d, axis=0).values
    internal_dist = torch.min(d_maxed, torch.zeros_like(d_maxed))

    return external_dist + internal_dist

def sdStar(p, r, n , m):
    '''
    https://iquilezles.org/articles/distfunctions2d/
    '''

    assert m >= 2 and m <= n
    an = np.pi / n
    en = np.pi / m
    acs = np.array([np.cos(an), np.sin(an)])[:, None]
    ecs = np.array([np.cos(en), np.sin(en)])[:, None]

    bn = np.arctan2(p[0], p[1])%(2 * an) - an

    p = np.linalg.norm(p, axis=0) * np.array([np.cos(bn), np.abs(np.sin(bn))])
    p -= r * acs
    p += ecs * np.clip(-np.dot(p.T, ecs), 0.0, r * acs[1]/ecs[1]).T
    return np.linalg.norm(p, axis=0) * np.sign(p[0])

def sdStarTorch(p,r,n,m):
    '''
        https://iquilezles.org/articles/distfunctions2d/
    '''
    an = np.pi / n
    en = np.pi / m

    acs = torch.tensor([math.cos(an), math.sin(an)])[:, None]
    ecs = torch.tensor([math.cos(en), math.sin(en)])[:, None]
    bn = torch.atan2(p[0], p[1])%(2 * an) - an

    bcs = torch.stack([torch.cos(bn), torch.abs(torch.sin(bn))])

    p = torch.norm(p, dim=0) * bcs 
    p_prime = p -  r * acs

    clamped_values = torch.clamp(-(p_prime.T @ ecs), 0.0, (r * acs[1]/ecs[1]).item()).T

    p_prime_prime = p_prime + ecs * clamped_values

    signed_dist = torch.norm(p_prime_prime, dim=0) * torch.sign(p_prime_prime[0])
    return signed_dist
    # mask = (signed_dist > 0)
    # print(an, en, acs, ecs, bn, bcs, p, p_prime, clamped_values, p_prime_prime, sep="\n")

#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    PTS = 10000
    # p = torch.rand(2, PTS) - 0.5
    p_np = np.random.random((2, PTS)) - 0.5
    p = torch.tensor(p_np).float()
    r = 0.5
    n = 5
    m = 3
    signed_dist_t = sdStarTorch(p, r, n, m)
    mask_t = (signed_dist_t > 0).detach().cpu()

    signed_dist = sdStar(p_np, r, n, m)
    mask = signed_dist > 0


    p_draw = p_np
    mask = mask_t

    fig, ax = plt.subplots(1,1,figsize=(4,4))
    ax.scatter(p_draw[0][mask], p_draw[1][mask])
    ax.set_aspect(1)
    plt.show()

#%%
