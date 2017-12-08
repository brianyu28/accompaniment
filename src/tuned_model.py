from util import normalize
import tuning


def most_likely_sequence(sequence, configuration):
    """
    Returns the most likely sequence of notes across all timesteps.
    Computed using Viterbi's Algorithm.
    """

    print("Computing most likely sequence of notes...")

    num_notes = len(sequence)
    num_states = num_notes + 1  # Add 1 for before all notes play.

    # Computed stores keys (time, state) and stores (prob, prev)
    # which are the probability of being at that state at the time
    # and the prior state
    computed = {}
    computed[0, 0] = (1, None)
    for i in range(1, num_states + 1):
        computed[0, i] = (0, None)

    # Tune parameters and get re-estimated emission and transition models
    seq_pitches, config_pitches = tuning.extract_pitches(sequence, configuration)
    emission_model, transition_model = tuning.tune_parameters(seq_pitches, config_pitches)

    def mls(time, state):
        """
        Computes most likely sequence at time up to a given state.
        Time t refers to sequence note index t - 1,
            since time 0 is before all notes.
        State s refers to configuration state index s - 1,
            since state 0 is before all notes.
        """
        if (time, state) in computed:
            return computed[time, state][0]

        evidence_prob = emission_model[seq_pitches[time - 1]][state]

        # Compute max over previous timestep.
        # Stores tuple of (state_number, probability)
        prevs = []
        for prev_state in range(len(configuration) + 1):
            p = transition_model[state][prev_state] * \
                mls(time - 1, prev_state)
            prevs.append((p, prev_state))
        prevs.sort(reverse=True)
        computed[time, state] = evidence_prob * prevs[0][0], prevs[0][1]
        return computed[time, state]

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
