import math
import numpy as np
import pickle
import base64

class MockScene():
    def __init__(self,
                    w_count=64,
                    h_count=44,
                    aspect_ratio = 0.7):
        self.w_count = w_count
        self.h_count = h_count
        self.robot_aspect_ratio = aspect_ratio

def sanitize_num(n):
    if math.isfinite(n):
        return n
    elif math.isnan(n):
        return "NaN"
    elif n == np.inf:
        return "Infinity"
    elif n == -np.inf:
        return "-Infinity"

def sanitize_list( l):
    return [validation_map[type(element)](element) for element in l]

def sanitize_genome( genome):
    for key, value in genome.items():
        genome[key] = validation_map[type(value)](value)
    return genome

validation_map = {dict: sanitize_genome,
                    type([]):sanitize_list,
                    type(1): sanitize_num,
                    type(0.1):sanitize_num}

def b64_encode(val):
    return base64.urlsafe_b64encode(pickle.dumps(val)).decode("utf-8")


def sanitize_result( result):
    result["loss"] = sanitize_num(result["loss"])
    assert result["loss"] == result["loss"]
    result["morphGenome"] = sanitize_genome(result["morphGenome"])
    result["actuationGenome"] = sanitize_genome(result["actuationGenome"])
    return result
