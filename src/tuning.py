from util import normalize
import numpy as np

def tune_parameters(sequence, configuration):
    """
    Runs the Baum-Welch algorithm to iteratively
    re-estimate the emission and transition probabilities
    of the Hidden Markov Model.
    """
    num_states = len(configuration) + 1
    num_obs = len(sequence) + 1

    # Initialize emission and transition probabilities according to our
    # knowledge of the domain
    emission_model = init_emissions(num_obs)
    transition_model = init_transitions(sequence, configuration, num_states)

    # old_emissions = None
    # old_transitions = None
    i = 0

    # Iterate until probabilities converge
    while i < 20:

        # Calculate forward and backward probabilities under current models
        forward_probs = forward_alg(sequence, configuration,
                                    emission_model, transition_model,
                                    num_states, num_obs)
        backward_probs = backward_alg(sequence, configuration,
                                      emission_model, transition_model,
                                      num_states, num_obs)

        # EXPECTATION STEP (E-STEP):

        # Calculate gamma for all x and t
        gammas = get_gammas(forward_probs, backward_probs, num_states, num_obs)

        # Calculate xi for all t, x_k, x_k+1
        xis = get_xis(forward_probs, backward_probs,
                      transition_model, emission_model,
                      num_states, num_obs)

        # MAXIMIZATION STEP (M-STEP):

        # Re-estimate transition probabilities
        old_emissions = emission_model
        emission_model = update_emission_model(sequence, gammas, num_states, num_obs)

        # Re-estimate emission probabilities
        old_transitions = transition_model
        transition_model = update_transition_model(gammas, xis, num_states)

    # Return optimized parameters
    return emission_model, transition_model

def compare_models(model1, model2): # for vectors, not matrices. Doesn't work
    """
    Compares two vectors by returning the dot product of their unit vectors.
    """
    unit1 = model1 / np.linalg.norm(model1)
    unit2 = model2 / np.linalg.norm(model2)
    return unit1.dot(unit2)

def init_emissions(num_obs):
    """
    Generates an initial emission model based on our knowledge of the domain.
    """
    emission_model = []
    for observed in xrange(128):
        for hidden in xrange(num_obs):

            # Default to each probability having equal weight.
            probs = [1 for i in range(128)]

            # Add weight to states within a range of 5 from the actual.
            for i in range(3):
                for j in range(max(0, hidden - i), min(128, hidden + i + 1)):
                    probs[j] += 10
        emission_model.append(normalize(probs))
    return emission_model

def init_transitions(sequence, configuration, num_states):
    """
    Generates an initial transition model based on our knowledge of the domain.
    """
    for cur_state in xrange(len(sequence)):
        for next_state in xrange(len(sequence)):
            probs = [0 for i in range(num_states)]

            # Descending probabilities for all future states.
            for i, j in enumerate(range(num_states - 1, cur_state, -1)):
                probs[j] = i + 1

            # Set probability for future and past.
            cur_sum = sum(probs)
            if cur_state != num_states - 1:
                probs[cur_state + 1] = cur_sum * 2
            probs[cur_state] = cur_sum * 1
    return transition_model

def forward_alg(sequence, configuration,
                emission_model, transition_model,
                num_states, num_obs):
    """
    Generates an array forward_probs through the forward algorithm,
    where forward_probs[t][cur_state] corresponds to the probability that
    cur_state occurs at time t, given all the evidence from time 1 to t.
    """
    num_states = len(configuration) + 1
    num_obs = len(sequence) + 1

    # Initialize base case according to initial distribution, which is that
    # the probability of being at note 0 is 100%.
    forward_probs = np.tile(np.repeat(None, num_states), (num_obs, 1))
    forward_probs[0] = np.repeat(0, num_states)
    forward_probs[0][0] = 1

    # Do forward algorithm calculations using current emission and transition
    # models, as well as the previously calculated forward probabilities.
    for t in xrange(1, num_obs + 1):
        for cur_state in xrange(num_states):
            temp = np.dot(forward_probs[t - 1], transition_model[cur_state])
            prob = temp * emission_model[sequence[t]][cur_state]
            forward_probs[t][cur_state] = prob
        normalize(forward_probs[t])

    # Return the array of forward probabilities.
    return forward_probs

def backward_alg(sequence, configuration,
                 emission_model, transition_model,
                 num_states, num_obs):
    """
    Generates an array backward_probs through the backward algorithm,
    where backward_probs[k + 1][cur_state] corresponds to
    the probability that the evidence from time k + 1 to t occurs,
    given that cur_state occurred at timestep k.
    """
    # Initialize base case for recursive calculation, which is that the
    # probability of emitting anything for the final timestep is equivalent to
    # the probability of transitioning from the penultimate to the final state.
    backward_probs = np.tile(np.repeat(None, num_states), (num_obs, 1))
    backward_probs[num_obs] = transition_model[sequence[num_obs]]
    backward_probs[0] = np.repeat(0, num_states)
    backward_probs[0][0] = 1

    # Run backward algorithm. The slight inefficiency of using k + 2 and k + 1
    # as indices instead of simply shifting k to start at num_obs or num_obs - 1
    # was chosen to make the code reflect the equations presented in lecture.
    for k in xrange(num_obs - 2, 0, -1):
        for cur_state in xrange(num_states):
            temp = backward_probs[k + 2] * transition_model[:, cur_state]
            prob = temp.dot(emission_model[sequence[k + 1]])
            backward_probs[k + 1][cur_state] = prob
        normalize(backward_probs[k + 1])

    return backward_probs

def get_gammas(forward_probs, backward_probs, num_states, num_obs):
    """
    Generates an array gammas, where gammas[k][cur_state] is
    the probability of being in state cur_state at time k
    given all observations from time 1 to t, where 1 < k < t.
    """
    gammas = np.tile(np.repeat(None, num_states), (num_obs, 1))

    for k in xrange(num_obs):
        gammas[k] = normalize(forward_probs[k] * backward_probs[k + 1])

    return gammas

def get_xis(forward_probs, backward_probs, transition_model, emission_model,
            num_states, num_obs):
    """
    Generates an array xis, where xis[k][cur_state][next_state]
    is the probability of being in cur_state at time k
    and next_state at time k + 1, where 1 < k < t,
    given all the observations from time 1 to t.
    """
    xis = np.tile(None, (num_states, num_states, num_obs))

    for k in xrange(num_obs):
        for cur_state in xrange(num_states):
            for next_state in xrange(num_states):
                prob = (forward_probs[k][cur_state][next_state]
                       * transition_model[next_state][cur_state]
                       * emission_model[sequence[k + 1]][next_state]
                       * backward_probs[k + 1][cur_state])
                xis[k][cur_state][next_state] = prob
            normalize(xis[k][cur_state])

    return xis

def update_emission_model(sequence, gammas, num_states, num_obs):
    """
    Returns the new emission model based on gamma.
    """
    emission_model = np.tile(np.repeat(None, num_states), (num_obs, 1))

    for observed in sequence:
        for i, cur_state in enumerate(configuration):
            num = np.sum(gammas[:, cur_state] * (sequence[i] == observed))
            denom = np.sum(gammas[:, cur_state])
            emission_model[observed][cur_state] = float(num) / denom

    return emission_model

def update_transition_model(gammas, xis, num_states):
    """
    Returns the new transition model based on gamma and xi.
    """
    transition_model = np.tile(np.repeat(None, num_states), (num_states, 1))

    for next_state in xrange(num_states):
        for cur_state in xrange(num_states):
            num = np.sum(xis[:, cur_state, next_state])
            denom = np.sum(gammas[:, cur_state])
            transition_model[next_state][cur_state] = float(num) / denom

    return transition_model
