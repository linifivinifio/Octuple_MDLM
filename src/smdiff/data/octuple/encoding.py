"""
OctupleMIDI Encoding/Decoding Module

This module implements the OctupleMIDI representation used in MusicBERT.
Each note is represented as an 8-tuple containing:
    (Bar, Position, Program, Pitch, Duration, Velocity, TimeSignature, Tempo)

The encoding quantizes continuous MIDI values into discrete tokens for use in
sequence models like transformers.
"""

import miditoolkit
import numpy as np
import math
import io

# ============================================================================
# CONSTANTS - Define the quantization and range parameters
# ============================================================================

pos_resolution = 16  # Quantization resolution per beat (16 positions per quarter note)
bar_max = 256  # Maximum number of bars to encode
velocity_quant = 4  # Velocity quantization step (128 velocities -> 32 bins)
tempo_quant = 12  # Tempo quantization using log scale (2 ** (1/12) steps)
min_tempo = 16  # Minimum tempo in BPM
max_tempo = 256  # Maximum tempo in BPM
duration_max = 8  # Maximum duration as 2^8 beats
max_ts_denominator = 6  # Maximum time signature denominator as 2^6 = 64
max_notes_per_bar = 2  # Maximum notes per bar for time signature encoding
beat_note_factor = 4  # MIDI standard: quarter note = 1 beat
trunc_pos = 2 ** 16  # Truncate position to ~30 minutes (1024 measures)
max_inst = 127  # Maximum MIDI instrument number
max_pitch = 127  # Maximum MIDI pitch
max_velocity = 127  # Maximum MIDI velocity

# ============================================================================
# TIME SIGNATURE ENCODING - Build lookup tables for time signature conversion
# ============================================================================

# Dictionary mapping (numerator, denominator) -> encoded index
ts_dict = dict()
# List mapping encoded index -> (numerator, denominator)
ts_list = list()

# Build all valid time signatures (e.g., 4/4, 3/4, 6/8, etc.)
for i in range(0, max_ts_denominator + 1):  # Denominators: 1, 2, 4, 8, 16, 32, 64
    for j in range(1, ((2 ** i) * max_notes_per_bar) + 1):  # Valid numerators for each denominator
        ts_dict[(j, 2 ** i)] = len(ts_dict)
        ts_list.append((j, 2 ** i))

# ============================================================================
# DURATION ENCODING - Build lookup tables for duration quantization
# ============================================================================

# dur_enc: maps raw duration (in ticks) -> encoded duration index
# dur_dec: maps encoded duration index -> decoded duration (in ticks)
dur_enc = list()
dur_dec = list()

# Build duration encoding with exponential quantization
# This allows representing both short and long durations efficiently
for i in range(duration_max):  # 0 to 7 (2^0 to 2^7 beats)
    for j in range(pos_resolution):  # 16 positions per beat
        dur_dec.append(len(dur_enc))
        for k in range(2 ** i):  # Repeat encoding for coarser quantization at longer durations
            dur_enc.append(len(dur_dec) - 1)

# ============================================================================
# ENCODING/DECODING HELPER FUNCTIONS
# ============================================================================

def t2e(x):
    """Time signature to encoded index (tuple -> int)"""
    assert x in ts_dict, 'unsupported time signature: ' + str(x)
    return ts_dict[x]

def e2t(x):
    """Encoded index to time signature (int -> tuple)"""
    if x >= len(ts_list):
        x = len(ts_list) - 1
    return ts_list[x]

def d2e(x):
    """Duration to encoded index (raw ticks -> int)"""
    return dur_enc[x] if x < len(dur_enc) else dur_enc[-1]

def e2d(x):
    """Encoded index to duration (int -> raw ticks)"""
    return dur_dec[x] if x < len(dur_dec) else dur_dec[-1]

def v2e(x):
    """Velocity to encoded index (0-127 -> 0-31)"""
    return x // velocity_quant

def e2v(x):
    """Encoded index to velocity (0-31 -> 0-127), returns midpoint of bin"""
    return (x * velocity_quant) + (velocity_quant // 2)

def b2e(x):
    """BPM (tempo) to encoded index using logarithmic scale"""
    x = max(x, min_tempo)  # Clamp to valid range
    x = min(x, max_tempo)
    x = x / min_tempo  # Normalize
    e = round(math.log2(x) * tempo_quant)  # Log scale quantization
    return e

def e2b(x):
    """Encoded index to BPM (tempo)"""
    return 2 ** (x / tempo_quant) * min_tempo

def time_signature_reduce(numerator, denominator):
    """
    Reduce time signature to fit within encoding constraints.
    
    1. Reduce denominator if too large (e.g., 4/128 -> 2/64)
    2. Decompose numerator if bar is too long (e.g., 12/4 -> 6/4)
    """
    # Reduction: when denominator is too large
    while denominator > 2 ** max_ts_denominator and denominator % 2 == 0 and numerator % 2 == 0:
        denominator //= 2
        numerator //= 2
    
    # Decomposition: when length of a bar exceeds max_notes_per_bar
    while numerator > max_notes_per_bar * denominator:
        for i in range(2, numerator + 1):
            if numerator % i == 0:
                numerator //= i
                break
    
    return numerator, denominator

# ============================================================================
# OCTUPLE ENCODING CLASS
# ============================================================================

class OctupleEncoding:
    """
    Helper class to convert MIDI files to OctupleMIDI tokens and vice versa.
    
    OctupleMIDI representation:
        Each note is encoded as an 8-tuple:
        (Bar, Position, Program, Pitch, Duration, Velocity, TimeSignature, Tempo)
        
    Where:
        - Bar: measure number (0 to bar_max-1)
        - Position: position within the bar (0 to measure_length-1)
        - Program: MIDI instrument (0-127 for melodic, 128 for percussion)
        - Pitch: MIDI pitch (0-127 for melodic, 128-255 for percussion)
        - Duration: quantized note duration
        - Velocity: quantized note velocity
        - TimeSignature: encoded time signature at this position
        - Tempo: encoded tempo at this position
    """
    
    def __init__(self):
        pass

    def encode_notesequence(self, note_sequence):
        """
        Converts a note_seq.NoteSequence to octuple tokens using in-memory streams.
        """
        # Change the import to get the PrettyMIDI converter directly
        from note_seq import note_sequence_to_pretty_midi
        import io
        
        # 1. Convert NoteSequence to a PrettyMIDI object first
        pm = note_sequence_to_pretty_midi(note_sequence)
        
        # 2. Create an in-memory buffer
        midi_stream = io.BytesIO()
        
        # 3. Write directly to the buffer 
        # (PrettyMIDI.write supports file-like objects, unlike the note_seq wrapper)
        pm.write(midi_stream)
        
        # 4. Rewind the buffer so it can be read
        midi_stream.seek(0)
        
        # 5. Pass the stream to your encode method
        return self.encode(midi_stream)

    def encode(self, input_obj):
        """
        Reads a MIDI file (or stream) and converts it to a sequence of Octuple tokens.
        
        Args:
            input_obj: Path to the MIDI file (str) or a file-like object (BytesIO)
            
        Returns:
            np.array of shape (N, 8)
        """
        # Load MIDI file
        try:
            # Handle both string paths and file-like objects
            if isinstance(input_obj, str):
                midi_obj = miditoolkit.MidiFile(input_obj)
            else:
                # Assuming input_obj is a file-like object (BytesIO)
                midi_obj = miditoolkit.MidiFile(file=input_obj)
        except Exception as e:
            print(f"Error loading MIDI input: {e}")
            return np.array([], dtype=int)

        # Helper function to convert MIDI ticks to quantized positions
        def time_to_pos(t):
            # Guard against zero division if ticks_per_beat is missing/zero
            tpb = midi_obj.ticks_per_beat if midi_obj.ticks_per_beat > 0 else 480
            return round(t * pos_resolution / tpb)

        # ... (The rest of the encode method remains exactly the same) ...
        # Start from: # Get all note start positions
        # notes_start_pos = [time_to_pos(j.start) ...
        
        # ... copy the rest of the original encode logic here ...
        
        # Ensure to include the rest of the function:
        # Get all note start positions
        notes_start_pos = [time_to_pos(j.start)
                           for i in midi_obj.instruments for j in i.notes]
        if len(notes_start_pos) == 0:
            return np.array([], dtype=int)

        # Determine the range of positions to encode
        max_pos = min(max(notes_start_pos) + 1, trunc_pos)
        
        # Create a lookup table: position -> [bar, time_sig, pos_in_bar, tempo]
        pos_to_info = [[None for _ in range(4)] for _ in range(max_pos)]
        
        # Extract time signature and tempo changes from MIDI
        tsc = midi_obj.time_signature_changes
        tpc = midi_obj.tempo_changes
        
        # Fill in time signatures for each position
        for i in range(len(tsc)):
            start_pos = time_to_pos(tsc[i].time)
            end_pos = time_to_pos(tsc[i + 1].time) if i < len(tsc) - 1 else max_pos
            for j in range(start_pos, end_pos):
                if j < len(pos_to_info):
                    pos_to_info[j][1] = t2e(time_signature_reduce(tsc[i].numerator, tsc[i].denominator))
        
        # Fill in tempos for each position
        for i in range(len(tpc)):
            start_pos = time_to_pos(tpc[i].time)
            end_pos = time_to_pos(tpc[i + 1].time) if i < len(tpc) - 1 else max_pos
            for j in range(start_pos, end_pos):
                if j < len(pos_to_info):
                    pos_to_info[j][3] = b2e(tpc[i].tempo)
        
        # Fill in default values
        for j in range(len(pos_to_info)):
            if pos_to_info[j][1] is None:
                pos_to_info[j][1] = t2e(time_signature_reduce(4, 4))
            if pos_to_info[j][3] is None:
                pos_to_info[j][3] = b2e(120.0)
        
        # Calculate bar numbers
        cnt = 0
        bar = 0
        measure_length = None
        
        for j in range(len(pos_to_info)):
            ts = e2t(pos_to_info[j][1])
            if cnt == 0:
                measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1]
            pos_to_info[j][0] = bar
            pos_to_info[j][2] = cnt
            cnt += 1
            if cnt >= measure_length:
                cnt -= measure_length
                bar += 1
        
        # Encode all notes
        encoding = []
        for inst in midi_obj.instruments:
            for note in inst.notes:
                if time_to_pos(note.start) >= trunc_pos:
                    continue
                
                info = pos_to_info[time_to_pos(note.start)]
                
                encoding.append((
                    info[0],  # Bar
                    info[2],  # Position within bar
                    max_inst + 1 if inst.is_drum else inst.program,
                    note.pitch + max_pitch + 1 if inst.is_drum else note.pitch,
                    d2e(time_to_pos(note.end) - time_to_pos(note.start)),
                    v2e(note.velocity),
                    info[1],  # Time signature
                    info[3]   # Tempo
                ))
        
        if len(encoding) == 0:
            return np.array([], dtype=int)
        
        encoding.sort()
        return np.array(encoding, dtype=int)

    def decode(self, tokens, return_notes=False):
        """
        Converts Octuple tokens back to a MIDI object.
        
        Args:
            tokens: list or np.array of shape (N, 8)
                Each row is [bar, position, program, pitch, duration, velocity, time_sig, tempo]
            return_notes: bool, if True, returns (midi_obj, notes_list)
                
        Returns:
            miditoolkit.MidiFile object (or tuple if return_notes=True)
        """
        if len(tokens) == 0:
            return miditoolkit.MidiFile()

        # Convert to list if numpy array
        encoding = tokens if isinstance(tokens, list) else tokens.tolist()
        
        # Filter out padding (all zeros) if any
        encoding = [x for x in encoding if not all(v == 0 for v in x)]
        if not encoding:
             return miditoolkit.MidiFile()

        # Step 1: Reconstruct time signatures for each bar
        # Collect all time signatures mentioned in each bar
        bar_to_timesig = [list() for _ in range(max(map(lambda x: x[0], encoding)) + 1)]
        for i in encoding:
            bar_to_timesig[i[0]].append(i[6])  # i[6] is the time signature
        
        # Use the most common time signature in each bar
        bar_to_timesig = [max(set(i), key=i.count) if len(i) > 0 else None for i in bar_to_timesig]
        
        # Fill in missing time signatures
        for i in range(len(bar_to_timesig)):
            if bar_to_timesig[i] is None:
                # Use 4/4 for first bar, otherwise use previous bar's time signature
                bar_to_timesig[i] = t2e(time_signature_reduce(4, 4)) if i == 0 else bar_to_timesig[i - 1]
        
        # Step 2: Calculate absolute position for each bar
        bar_to_pos = [None] * len(bar_to_timesig)
        cur_pos = 0
        
        for i in range(len(bar_to_pos)):
            bar_to_pos[i] = cur_pos
            ts = e2t(bar_to_timesig[i])
            # Calculate measure length based on time signature
            measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1]
            cur_pos += measure_length
        
        # Step 3: Reconstruct tempo for each position
        # Collect all tempos mentioned at each position
        pos_to_tempo = [list() for _ in range(cur_pos + max(map(lambda x: x[1], encoding)) + 1)]
        for i in encoding:
            absolute_pos = bar_to_pos[i[0]] + i[1]
            pos_to_tempo[absolute_pos].append(i[7])  # i[7] is the tempo
        
        # Average tempos at each position
        pos_to_tempo = [round(sum(i) / len(i)) if len(i) > 0 else None for i in pos_to_tempo]
        
        # Fill in missing tempos
        for i in range(len(pos_to_tempo)):
            if pos_to_tempo[i] is None:
                # Use 120 BPM for first position, otherwise use previous position's tempo
                pos_to_tempo[i] = b2e(120.0) if i == 0 else pos_to_tempo[i - 1]
        
        # Step 4: Create MIDI object
        midi_obj = miditoolkit.MidiFile()
        
        # Helper function to convert (bar, position) to MIDI ticks
        def get_tick(bar, pos):
            return (bar_to_pos[bar] + pos) * midi_obj.ticks_per_beat // pos_resolution
        
        # Create all possible instruments (0-127 melodic + 128 percussion)
        midi_obj.instruments = [miditoolkit.Instrument(
            program=(0 if i == 128 else i),
            is_drum=(i == 128),
            name=str(i)
        ) for i in range(128 + 1)]
        
        # Step 5: Add all notes
        all_notes = []
        for i in encoding:
            start = get_tick(i[0], i[1])  # Convert to MIDI ticks
            program = i[2]
            
            # Decode pitch (subtract offset for percussion)
            if program == 128:
                pitch = i[3] - 128
            else:
                pitch = i[3]
            
            # Clamp pitch to valid MIDI range
            pitch = max(0, min(127, pitch))
            
            # Decode duration
            duration = get_tick(0, e2d(i[4]))
            if duration == 0:
                duration = 1  # Ensure minimum duration
            end = start + duration
            
            # Decode velocity
            velocity = e2v(i[5])
            # Clamp velocity to valid MIDI range
            velocity = max(0, min(127, velocity))
            
            note = miditoolkit.Note(start=start, end=end, pitch=pitch, velocity=velocity)
            all_notes.append(note)
            
            # Add note to the appropriate instrument
            midi_obj.instruments[program].notes.append(note)
        
        # Remove empty instruments
        midi_obj.instruments = [i for i in midi_obj.instruments if len(i.notes) > 0]
        
        # Step 6: Add time signature changes
        cur_ts = None
        for i in range(len(bar_to_timesig)):
            new_ts = bar_to_timesig[i]
            if new_ts != cur_ts:  # Only add when time signature changes
                numerator, denominator = e2t(new_ts)
                midi_obj.time_signature_changes.append(
                    miditoolkit.TimeSignature(
                        numerator=numerator,
                        denominator=denominator,
                        time=get_tick(i, 0)
                    )
                )
                cur_ts = new_ts
        
        # Step 7: Add tempo changes
        cur_tp = None
        for i in range(len(pos_to_tempo)):
            new_tp = pos_to_tempo[i]
            if new_tp != cur_tp:  # Only add when tempo changes
                tempo = e2b(new_tp)
                midi_obj.tempo_changes.append(
                    miditoolkit.TempoChange(tempo=tempo, time=get_tick(0, i))
                )
                cur_tp = new_tp
        
        # Step 8: Calculate and set max_tick
        # This is required for miditoolkit's get_tick_to_time_mapping() to work correctly
        if all_notes:
            midi_obj.max_tick = max(note.end for note in all_notes)
        else:
            midi_obj.max_tick = 0
        
        if return_notes:
            return midi_obj, all_notes
        return midi_obj
