

def normalize(lst):
    """
    Normalize a probability distribution.
    """
    return [float(i) / sum(lst) for i in lst]
