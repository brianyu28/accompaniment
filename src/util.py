

def normalize(lst):
    """
    Normalize a probability distribution.
    """
    if sum(lst) == 0.0:
        return lst
    else:
        return [float(i) / sum(lst) for i in lst]
