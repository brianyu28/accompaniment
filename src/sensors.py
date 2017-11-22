import sys

from aubio import source, notes


def main():
    """
    Extracts notes from a file.
    When called as a command-line script, just outputs the notes.
    """
    notes = extract_notes(sys.argv[1])
    for note in notes["notes"]:
        print(note)

def extract_notes(filename):
    """
    Takes a .wav file and uses Fast Fourier Transform
    to convert to notes.

    Adapted from
    https://github.com/aubio/aubio/blob/master/python/demos/demo_notes.py
    """

    # Define constants for audio extraction.
    downsample = 1
    samplerate = 44100 // downsample
    win_s = 512 // downsample  # FFT size
    hop_s = 256 // downsample  # hop size

    # Set up aubio source reading.
    s = source(filename, samplerate, hop_s)
    samplerate = s.samplerate
    tolerance = 0.8
    notes_o = notes("default", win_s, hop_s, samplerate)
    total_frames = 0
    sequence = []

    # Extract notes.
    while True:
        samples, read = s()
        new_note = notes_o(samples)
        if (new_note[0] != 0):
            time = total_frames / float(samplerate)
            sequence.append((time, new_note[0], new_note[1]))
        total_frames += read
        if read < hop_s:
            break
    return {
        "notes": sequence,
        "duration": total_frames / float(samplerate)
    }

if __name__ == "__main__":
    main()
