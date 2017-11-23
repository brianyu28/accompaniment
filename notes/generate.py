from midiutil import MIDIFile

MyMIDI = MIDIFile(1, adjust_origin=True)
track = 0
channel = 0
pitch = 60
time = 0
duration = 1
volume = 100
MyMIDI.addNote(track,channel,pitch,time,duration,volume)
