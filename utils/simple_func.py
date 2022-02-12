import numpy as np

def list_to_str(data):
    result = []
    for c in data:
        result.append("".join(c))
    result = np.array(result)
    return result