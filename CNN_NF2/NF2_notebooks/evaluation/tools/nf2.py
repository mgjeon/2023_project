import json
from nf2.evaluation.unpack import load_cube

def load_b_from_nf2_file(nf2_file):
    b = load_cube(nf2_file, progress=True)
    return b
