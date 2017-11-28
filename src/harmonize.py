import json
import sys

import generator
import model
import sensors
import volume


def main():

    # Parse command line arguments.
    try:
        _, config, filename = sys.argv
    except ValueError:
        print("Usage: harmonize config.json filename.wav")
        sys.exit(1)

    # Read data from input files.
    try:
        sequence = sensors.extract_notes(filename)
        configuration = json.loads(open(config).read())
    except RuntimeError:
        print("Error parsing input file.")
        sys.exit(2)
    except FileNotFoundError:
        print("Configuration file could not be found.")
        sys.exit(3)
    except (json.decoder.JSONDecodeError, ValueError):
        print("Error parsing configuration file.")
        sys.exit(4)

    # Compute the most likely sequence.
    mls = model.most_likely_sequence(sequence["notes"], configuration["piece"])
    volumes = volume.get_volumes(configuration, sequence["notes"], mls)
    generator.generate(filename, sequence, configuration, mls, volumes)


if __name__ == "__main__":
    main()
