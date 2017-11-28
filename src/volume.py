from random import randrange
from statistics import median


def get_volumes(configuration, sequence, mls):
    """
    Given a configuration piece and a sequence of notes
    with their corresponding velocities, performs k-means
    clustering to identify the likely volumes of the 
    dynamic symbols in the piece.
    """
    played = set()

    # Get only the notes we think are actually notes.
    data = []
    for i, note in enumerate(mls):
        if note not in played:
            played.add(note)
            data.append(sequence[i][2])

    # Get the set of all dynamics.
    dynamics = sorted(list(set(note["vel"] for note in configuration["piece"])))
    means = k_means(len(dynamics), data)
    return {dynamics[i]: means[i] for i in range(len(dynamics))}


def k_means(k, data):
    """
    Performs a variant on k-means clustering to learn
    dynamics data.

    In particular, uses dynamic data from the piece, and finds a
    local optimum for minimizing the distance to means.
    Then, deviating from the original k-means algorithm,
    takes the median of the cluster, to avoid being skewed
    too much by outlier data points.
    """

    # Generate k means.
    data_range = min(data), max(data)
    means = set()
    while True:
        if len(means) == k:
            break
        means.add(randrange(data_range[0], data_range[1]))
    means = list(means)

    # Make assignments.
    assignments = [None for i in range(len(data))]

    # Repeat the algorithm.
    while True:

        # Assign each point to nearest mean.
        changed = False
        for i, point in enumerate(data):
            dists = [abs(point - mean) for mean in means]
            new_assignment = dists.index(min(dists))
            if assignments[i] != new_assignment:
                changed = True
                assignments[i] = new_assignment

        # Recalculate means.
        for i in range(len(means)):
            c = cluster(i, data, assignments)
            if len(c) == 0:
                new_mean = randrange(data_range[0], data_range[1])
            else:
                new_mean = sum(c) / len(c)
            means[i] = new_mean


        if not changed:
            break

    # Take the median of each resulting cluster as our velocity.
    for i in range(len(means)):
        c = cluster(i, data, assignments)
        means[i] = median(c)
    means = [int(i) for i in sorted(means)]
    return means


def cluster(i, data, assignments):
    """
    Returns all data in the data array in cluster i,
    where data is assigned based on assignments.
    """
    return [data[j] for j in range(len(data)) if assignments[j] == i]
