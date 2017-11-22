from util import normalize

def most_likely_sequence(sequence, configuration):
    """
    Returns the most likely sequence of notes across all timesteps.
    Computed using Viterbi's Algorithm.
    """

    num_notes = len(sequence)
    num_states = num_notes + 1  # Add 1 for before all notes play.

    # Computed stores keys (time, state) and stores (prob, prev)
    # which are the probability of being at that state at the time
    # and the prior state
    computed = {}
    computed[0, 0] = (1, None)
    for i in range(1, num_states + 1):
        computed[0, i] = (0, None)

    def mls(time, state):
        """
        Computes most likely sequence at time up to a given state.
        Time t refers to sequence note index t - 1, since time 0 is before all notes.
        State s refers to configuration state index s - 1, since state 0 is before all notes.
        """
        if (time, state) in computed:
            return computed[time, state][0]
        
        evidence_prob = emission(sequence[time - 1][1], configuration[state - 1]["pitch"])

        # Compute max over previous timestep.
        # Stores tuple of (state_number, probability)
        prevs = []
        for prev_state in range(len(configuration) + 1):
            p = transition(state, prev_state, num_states) * mls(time - 1, prev_state)
            prevs.append((p, prev_state))
        prevs.sort(reverse=True)
        computed[time, state] = prevs[0]
        return prevs[0]
    
    # Compute all most likely sequences.
    for t in range(1, len(sequence) + 1):
        for s in range(0, len(configuration) + 1):
            mls(t, s)
    
    # Extract states: list of states at each timestep.
    states = []
    cur_time = len(sequence)
    cur_state = len(configuration)
    while cur_time >= 0:
        states.insert(0, cur_state)
        _, next_state = computed[cur_time, cur_state]
        cur_state = next_state
        cur_time -= 1
    
    return states 


def emission(observed, hidden):
    """
    Returns the emission probability
    of seeing a particular observed note
    given a hidden state.
    """
    observed = int(observed)
    hidden = int(hidden)

    # Default to each probability having equal weight.
    probs = [1 for i in range(128)]

    # Add weight to states within a range of 5 from the actual.
    for i in range(3):
        for j in range(max(0, hidden - i), min(128, hidden + i + 1)):
            probs[j] += 10

    # Normalize and return the actual probability.
    return normalize(probs)[observed]

def transition(next_state, cur_state, num_states):
    """
    Returns the transition probability of
    going to the next state given the current state.
    """
    probs = [0 for i in range(num_states)]

    # Descending probabilities for all future states.
    for i, j in enumerate(range(num_states - 1, cur_state, -1)):
        probs[j] = i + 1
    
    # Set probability for future and past.
    cur_sum = sum(probs)
    if cur_state != num_states  - 1:
        probs[cur_state + 1] = cur_sum * 3
    probs[cur_state] = cur_sum * 3
    return normalize(probs)[next_state]
