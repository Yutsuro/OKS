import numpy as np

from decimal import Decimal, ROUND_HALF_UP


def edit_keypoints(kpts):
    kpts = np.array(kpts).reshape(-1,3)
    vi = kpts[:,2]
    kpts = kpts[:,0:2]
    return kpts, vi


def OKS(kpts1, kpts2, sigma, area):

    kpts1, vi1 = edit_keypoints(kpts1)
    kpts2, vi2 = edit_keypoints(kpts2)

    if np.shape(kpts1) != np.shape(kpts2):
        print(kpts1, kpts2)
        print(np.shape(kpts1), np.shape(kpts2))
        raise ValueError("not same size")
    
    k = 2*sigma
    s = area

    d = np.linalg.norm(kpts1 - kpts2, ord=2, axis=1)
    v = np.ones(len(d))

    for part in range(len(d)):
        if vi1[part] == 0 or vi2[part] == 0:
            d[part] = 0
            v[part] = 0
    
    if np.sum(v)!=0:
        OKS = (np.sum([(np.exp((-d[i]**2)/(2*s*(k[i]**2))))*v[i] for i in range(len(d))])/np.sum(v))
    else:
        OKS = 0
    
    OKS = float(Decimal(str(OKS)).quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP))

    return OKS