import os
import uuid

from contextlib import redirect_stdout
from midi2audio import FluidSynth
from midiutil import MIDIFile
from pydub import AudioSegment


def generate(filename, sequence, configuration, mls, volumes):
    """
    Generates merged audio file.
    """
    print("Generating accompaniment audio...")

    # Maintain dictionary of which notes of configuration to play.
    notes = {0: None}
    prev = None
    for i, note_number in enumerate(mls):

        # Don't play notes we've already played, or that aren't meant to be.
        if note_number in notes or \
           str(note_number) not in configuration["accompaniment"]:
            continue

        # Add note to dictionary.
        time = sequence["notes"][i - 1][0]
        velocity = volumes[configuration["piece"][note_number - 1]["vel"]]
        pitches = configuration["accompaniment"][str(note_number)]
        notes[note_number] = {
            "time": time,
            "velocity": velocity,
            "pitches": pitches
        }

        # Set duration of the prior note.
        if prev is not None:
            notes[prev]["duration"] = time - notes[prev]["time"]
        prev = note_number

    # Set duration of the final note.
    notes[prev]["duration"] = sequence["duration"] - notes[prev]["time"]

    # Configure MIDI file.
    midifile = MIDIFile(1, adjust_origin=True)
    track, channel = 0, 0
    midifile.addTempo(0, 0, 60)

    # Add notes to MIDI file.
    for i in range(1, prev + 1):
        if i not in notes:
            continue
        for pitch in notes[i]["pitches"]:
            midifile.addNote(track, channel, pitch,
                             notes[i]["time"], notes[i]["duration"], notes[i]["velocity"])

    # Create temporary MIDI file.
    identifier = str(uuid.uuid4())
    with open("{}.mid".format(identifier), "wb") as outfile:
        midifile.writeFile(outfile)

    # Convert temporary MIDI file to teporary WAV file.
    FluidSynth().midi_to_audio("{}.mid".format(identifier),
                               "{}.wav".format(identifier))

    # Combine accompaniment audio with soloist audio.
    merge(filename, "{}.wav".format(identifier), "output.wav")

    # Clean up temporary files.
    os.remove("{}.mid".format(identifier))
    os.remove("{}.wav".format(identifier))


def merge(file1, file2, outfile):
    """
    Overlays two .wav files and converts to a single .wav output.
    """
    print("Merging recording with accompaniment...")

    # Take .wav file inputs and adjust volume.
    audio1 = AudioSegment.from_file(file1)
    audio1 = audio1 - 10
    audio2 = AudioSegment.from_file(file2)
    audio2 = audio2 + 10

    # Overlay audio and export to file.
    overlayed = audio1.overlay(audio2)
    overlayed.export(outfile, format="wav")
