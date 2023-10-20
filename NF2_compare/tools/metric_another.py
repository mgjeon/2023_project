import numpy as np

def vector_norm(vector):
    return np.linalg.norm(vector, axis=-1)

def dot_product(a, b):
    return (a*b).sum(-1)


def metrics(b, B):
    # b : model solution
    # B : reference magnetic field
    eps = 1e-7

    result = {}

    result['c_vec'] = np.sum(dot_product(B, b)) / np.sqrt(np.sum(vector_norm(B)**2) * np.sum(vector_norm(b)**2))
    
    M = np.prod(B.shape[:-1])
    eps = 1e-7
    nu = dot_product(B, b)
    de = vector_norm(B) * vector_norm(b)
    result['c_cs'] = (1 / M) * np.sum(np.divide(nu, de, where=de!=0.))
    result['c_cs_ep'] = (1 / M) * np.sum(nu/(de + eps))

    E_n = np.sum(vector_norm(b - B)) / np.sum(vector_norm(B))
    result["E_n'"] = 1 - E_n
    
    nu = vector_norm(b - B)
    de = vector_norm(B)
    E_m = (1 / M) * np.sum(np.divide(nu, de, where=de!=0.))
    result["E_m'"] = 1 - E_m
    E_m = (1 / M) * np.sum((nu/(de + eps)))
    result["E_m'_ep"] = 1 - E_m

    result['eps'] = np.sum(vector_norm(b)**2) / np.sum(vector_norm(B)**2)

    return result
