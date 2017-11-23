import os
import uuid

from contextlib import redirect_stdout
from midi2audio import FluidSynth
from midiutil import MIDIFile
from pydub import AudioSegment


def generate(filename, sequence, configuration, mls):
    """
    Generates merged audio file.
    """
    notes = {0: None}
    prev = None
    for i, note_number in enumerate(mls):
        if note_number in notes or \
            str(note_number) not in configuration["accompaniment"]:
            continue
        time = sequence["notes"][i - 1][0]
        pitch = configuration["accompaniment"][str(note_number)]["pitch"]
        notes[note_number] = {
            "time": time,
            "pitch": pitch
        }
        if prev is not None:
            notes[prev]["duration"] = time - notes[prev]["time"]
        prev = note_number
    notes[prev]["duration"] = sequence["duration"] - notes[prev]["time"]

    # Add notes to MIDI file.
    midifile = MIDIFile(1, adjust_origin=True)
    track, channel = 0, 0
    midifile.addTempo(0, 0, 60)
    for i in range(1, prev + 1):
        if i not in notes:
            continue
        midifile.addNote(track, channel, notes[i]["pitch"],
                notes[i]["time"], notes[i]["duration"], 100)

    # Create temporary MIDI file.
    identifier = str(uuid.uuid4())
    with open("{}.mid".format(identifier), "wb") as outfile:
        midifile.writeFile(outfile)

    # Conver temporary MIDI file to teporary WAV file.
    FluidSynth().midi_to_audio("{}.mid".format(identifier),
                               "{}.wav".format(identifier))
    merge(filename, "{}.wav".format(identifier), "output.wav")
    os.remove("{}.mid".format(identifier))
    os.remove("{}.wav".format(identifier))


def merge(file1, file2, outfile):
    """
    Overlays two .wav files and converts to a single .wav output.
    """
    audio1 = AudioSegment.from_file(file1)
    audio1 = audio1 - 10
    audio2 = AudioSegment.from_file(file2)
    audio2 = audio2 + 10
    overlayed = audio1.overlay(audio2)
    overlayed.export(outfile, format="wav")
