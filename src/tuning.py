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

    seq_pitches, config_pitches = extract_pitches(sequence, configuration)
    emission_model, transition_model = tune_parameters(seq_pitches, config_pitches)

def extract_pitches(sequence, configuration):
    seq_pitches = [int(i[1]) for i in sequence["notes"]]
    config_pitches = [i["pitch"] for i in configuration["piece"]]
    return seq_pitches, config_pitches

def tune_parameters(sequence, configuration):
    """
    Runs the Baum-Welch algorithm to iteratively
    re-estimate the emission and transition probabilities
    of the Hidden Markov Model.
    """
    num_states = len(configuration) + 1
    num_obs = len(sequence) + 1
    # print("Num_states")
    # print(num_states)
    #
    # print("Num_obs")
    # print(num_obs)

    print("Sequence:")
    print(sequence)
    print("Configuration")
    print(configuration)

    # Initialize emission and transition probabilities according to our
    # knowledge of the domain
    emission_model = init_emissions(num_obs, num_states, sequence, configuration)
    transition_model = init_transitions(sequence, configuration, num_states)
    print("Initial transition model")
    print(transition_model)
    print("Initial emission model")
    print(emission_model)

    # old_emissions = None
    # old_transitions = None
    i = 0

    print("Initiating tuning...")
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
    print(emission_model)
    print(transition_model)
    return emission_model, transition_model

def compare_models(model1, model2): # for vectors, not matrices. Doesn't work
    """
    Compares two vectors by returning the dot product of their unit vectors.
    """
    unit1 = model1 / np.linalg.norm(model1)
    unit2 = model2 / np.linalg.norm(model2)
    return unit1.dot(unit2)

def init_emissions(num_obs, num_states, sequence, configuration):
    """
    Generates an initial emission model based on our knowledge of the domain.
    """
    # Default to each probability having equal weight.
    emission_model = np.tile(np.repeat(1.0, 128), (128, 1))

    for i in xrange(128):
        emission_model[i] = normalize(emission_model[i])

    return emission_model

    for observed in sequence:
        # Add weight to states within a range of 5 from the actual.
        for i in range(3):
            for j in range(max(0, observed - i), min(128, observed + i + 1)):
                emission_model[observed][j] += 10
        # print("before")
        # print(emission_model[observed])
        # print("normalizing emission[observed]")
        emission_model[observed] = normalize(emission_model[observed])
        # print(emission_model[observed])
    return emission_model

def init_transitions(sequence, configuration, num_states):
    """
    Generates an initial transition model based on our knowledge of the domain.
    """
    transition_model = np.tile(np.repeat(1.0, num_states), (num_states, 1))

    for i in xrange(num_states):
        transition_model[i] = normalize(transition_model[i])
    return transition_model

    for cur_state in xrange(num_states):
        for next_state in xrange(num_states):
            probs = [0 for i in range(num_states)]

            # Descending probabilities for all future states.
            for i, j in enumerate(range(num_states - 1, cur_state, -1)):
                probs[j] = i + 1

            # Set probability for future and past.
            cur_sum = sum(probs)
            if cur_state != num_states - 1:
                probs[cur_state + 1] = cur_sum * 2
            probs[cur_state] = cur_sum * 1

            transition_model[cur_state][next_state] = probs[cur_state]
            print("Transition model row")
            print(transition_model[cur_state])
        transition_model[cur_state] = normalize(transition_model[cur_state])
    print("Transposing")
    print(transition_model.transpose())
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
    forward_probs[0] = np.repeat(0, num_states)
    forward_probs[0][0] = 1

    # Do forward algorithm calculations using current emission and transition
    # models, as well as the previously calculated forward probabilities.
    for t in xrange(1, num_obs + 1):
        for cur_state in xrange(num_states):
            temp = np.dot(forward_probs[t - 1], transition_model[cur_state])
            prob = temp * emission_model[sequence[t]][configuration[cur_state]]
            forward_probs[t][cur_state] = prob
        try:
            normalize(forward_probs[t])
        except:
            pass

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

if __name__ == "__main__":
    main()
