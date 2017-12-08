"""
To use the tuned parameters, change line 5 in harmonizer.py from
import model
to
import tuned_model as model
"""

from util import normalize
import numpy as np
import sys
import sensors
import json

def main():
    # Parse command line arguments.
    try:
        _, config, filename = sys.argv
    except ValueError:
        print("Usage: tuning config.json filename.wav")
        sys.exit(1)

    # Read data from input files.
    try:
        sequence = sensors.extract_notes(filename)
        configuration = json.loads(open(config).read())
    except RuntimeError:
        print("Error parsing input file.")
        sys.exit(2)
    except IOError:
        print("Configuration file could not be found.")
        sys.exit(3)
    except (json.decoder.JSONDecodeError, ValueError):
        print("Error parsing configuration file.")
        sys.exit(4)

    # Extract midi note numbers from sequence and configuration
    seq_pitches, config_pitches = extract_pitches(sequence["notes"],
                                                  configuration["piece"])
    # Tune parameters and get updated emission and transition models
    emission_model, transition_model = tune_parameters(seq_pitches,
                                                       config_pitches)

def extract_pitches(sequence, configuration):
    """
    Extracts midi note numbers from sequence and configuration,
    returns them as lists.
    """
    seq_pitches = [int(i[1]) for i in sequence]
    config_pitches = [i["pitch"] for i in configuration]
    return seq_pitches, config_pitches

def tune_parameters(sequence, configuration):
    """
    Runs the Baum-Welch algorithm to iteratively
    re-estimate the emission and transition probabilities
    of the Hidden Markov Model.
    """
    num_states = len(configuration) + 1
    num_obs = len(sequence) + 1
    full_sequence = [0] + sequence
    full_config = [0] + configuration

    # Initialize emission and transition probabilities according to our
    # knowledge of the domain
    emission_model = init_emissions(num_obs, num_states, sequence, full_config)
    transition_model = init_transitions(sequence, configuration, num_states)

    i = 0
    print("Initiating tuning...")
    while i < 30:
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

        # Calculate xi for all t, cur_state, next_state
        xis = get_xis(forward_probs, backward_probs,
                      transition_model, emission_model,
                      num_states, num_obs, sequence)

        # MAXIMIZATION STEP (M-STEP):

        # Re-estimate emission probabilities
        old_emissions = emission_model
        emission_model = update_emission_model(full_sequence, configuration,
                                               gammas, num_states, num_obs)

        # Re-estimate transition probabilities
        old_transitions = transition_model
        transition_model = update_transition_model(gammas, xis, num_states)

        i += 1

    # Return optimized parameters
    return emission_model, transition_model

def init_emissions(num_obs, num_states, sequence, configuration):
    """
    Generates an initial emission model based on our knowledge of the domain.
    """
    # Default to each probability having equal weight.
    emission_model = np.tile(np.repeat(1.0, num_states), (128, 1))

    for cur_state in range(num_states):
        # Add weight to states within a range of 5 from the actual.
        midi_num = configuration[cur_state]
        for i in range(3):
            for j in range(max(0, midi_num - i), min(128, midi_num + i + 1)):
                emission_model[j][cur_state] += 10

    # Make weights for cur_state 0 (no note playing) uniform
    emission_model[:, 0] = np.repeat(1.0, 128)
    for i in range(num_states):
        emission_model[:,i] = normalize(emission_model[:,i])

    return emission_model

def init_transitions(sequence, configuration, num_states):
    """
    Generates an initial transition model based on our knowledge of the domain.
    """
    # Default to each probability having equal weight
    transition_model = np.tile(np.repeat(1.0, num_states), (num_states, 1))

    for cur_state in range(num_states):
        for next_state in range(num_states):
            probs = [0 for i in range(num_states)]

            # Descending probabilities for all future states.
            for i, j in enumerate(range(num_states - 1, cur_state, -1)):
                probs[j] = i + 1

            # Set probability for future and past.
            cur_sum = sum(probs)
            if cur_state != num_states - 1:
                probs[cur_state + 1] = cur_sum * 2
            probs[cur_state] = cur_sum * 1

        # Normalize probabilities
        transition_model[cur_state] = normalize(probs)

    # Transpose array so that row indices correspond to next_state,
    # column indices correspond to cur_state
    return transition_model.transpose()

def forward_alg(sequence, configuration,
                emission_model, transition_model,
                num_states, num_obs):
    """
    Generates an array forward_probs through the forward algorithm,
    where forward_probs[t][cur_state] corresponds to the probability that
    cur_state occurs at time t, given all the evidence from time 1 to t.
    """
    # Initialize base case according to initial distribution, which is that
    # the probability of being at note 0 is 100%.
    forward_probs = np.tile(np.repeat(None, num_states), (num_obs, 1))
    forward_probs[0] = np.repeat(0.0, num_states)
    forward_probs[0][0] = 1.0

    # Do forward algorithm calculations using current emission and transition
    # models, as well as the previously calculated forward probabilities.
    for t in range(1, num_obs):
        for cur_state in range(num_states):
            temp = np.dot(forward_probs[t - 1], transition_model[cur_state])
            cur_obs = sequence[t - 1]
            prob = temp * emission_model[cur_obs][cur_state]
            forward_probs[t][cur_state] = prob
        forward_probs[t] = normalize(forward_probs[t])

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
    backward_probs[num_obs - 1] = transition_model[num_states - 1]
    backward_probs[0] = np.repeat(0.0, num_states)
    backward_probs[0][0] = 1.0

    # Run backward algorithm. The slight inefficiency of using k + 2 and k + 1
    # as indices instead of simply shifting k to start at num_obs or num_obs - 1
    # was chosen to make the code reflect the equations presented in lecture.
    for k in range(num_obs - 3, -1, -1):
        for cur_state in range(num_states):
            temp = backward_probs[k + 2] * transition_model[:, cur_state]
            prob = temp.dot(emission_model[sequence[k]])
            backward_probs[k + 1][cur_state] = prob
        backward_probs[k + 1] = normalize(backward_probs[k + 1])

    return backward_probs

def get_gammas(forward_probs, backward_probs, num_states, num_obs):
    """
    Generates an array gammas, where gammas[k][cur_state] is
    the probability of being in state cur_state at time k
    given all observations from time 1 to t, where 1 < k < t.
    """
    gammas = np.tile(np.repeat(None, num_states), (num_obs, 1))

    # Calculate gammas from forward and backwards probabilities
    for k in range(num_obs - 1):
        gammas[k] = normalize(forward_probs[k] * backward_probs[k + 1])
        
    # Gammas for last timestep are equal to forward probabilities
    gammas[num_obs - 1] = forward_probs[num_obs - 1]

    return gammas

def get_xis(forward_probs, backward_probs, transition_model, emission_model,
            num_states, num_obs, sequence):
    """
    Generates an array xis, where xis[k][cur_state][next_state]
    is the probability of being in cur_state at time k
    and next_state at time k + 1, where 1 < k < t,
    given all the observations from time 1 to t.
    """
    xis = np.tile(None, (num_obs, num_states, num_states))

    # Calculate xis from current emission and transition models
    # as well as the forward and backward probabilities.
    for k in range(num_obs):
        for cur_state in range(num_states):
            for next_state in range(num_states):
                prob = (forward_probs[k][cur_state]
                       * transition_model[next_state][cur_state])
                if k == 0:
                    prob = prob * emission_model[0][next_state]
                else:
                    prob = prob * emission_model[sequence[k - 1]][next_state]
                if k < (num_obs - 1):
                    prob = prob * backward_probs[k + 1][cur_state]
                xis[k][cur_state][next_state] = prob
            xis[k][cur_state] = normalize(xis[k][cur_state])

    return xis

def update_emission_model(sequence, configuration, gammas, num_states, num_obs):
    """
    Returns the new emission model based on gamma.
    """
    # Initiate emission model with probabilities having equal non-zero weight
    emission_model = np.tile(np.repeat(0.1, num_states), (128, 1))

    for observed in sequence:
        for cur_state in range(num_states):
            # Numerator is expected number of times
            # observed is emitted from cur_state
            num = np.sum(gammas[:, cur_state] * (sequence[cur_state] == observed))
            # Denominator is expected number of total emissions from cur_state
            denom = np.sum(gammas[:, cur_state])
            if denom != 0.0:
                # Re-estimated emission probability is the ratio
                emission_model[observed][cur_state] += float(num) / denom

    # Normalize probabilities
    for i in range(128):
        emission_model[i] = normalize(emission_model[i])

    return emission_model

def update_transition_model(gammas, xis, num_states):
    """
    Returns the new transition model based on gamma and xi.
    """
    # Initiate emission model with probabilities having equal non-zero weight
    transition_model = np.tile(np.repeat(0.1, num_states), (num_states, 1))

    for next_state in range(num_states):
        for cur_state in range(num_states):
            # Numerator is the expected number of times
            # next_state follows cur_state
            num = np.sum(xis[:, cur_state, next_state])
            # Denominator is the expected number of times
            # there is a transition out of cur_state
            denom = np.sum(gammas[:, cur_state])
            if denom != 0.0:
                # Re-estimated transition probability is the ratio
                transition_model[next_state][cur_state] += float(num) / denom
        transition_model[next_state] = normalize(transition_model[next_state])

    return transition_model

if __name__ == "__main__":
    main()
