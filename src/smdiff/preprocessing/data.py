# Copyright 2022 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MusicVAE data library."""
import abc
import collections
import copy
import functools
import itertools
import os
import tempfile

import note_seq
import numpy as np
from note_seq import chords_lib
from note_seq import drums_encoder_decoder
from note_seq import sequences_lib

from . import drum_pipelines
from . import melody_pipelines
from ..data.octuple import OctupleEncoding

PIANO_MIN_MIDI_PITCH = 21 # 21 is the lowest MIDI pitch for Pianos
PIANO_MAX_MIDI_PITCH = 108 # 108 is the highest MIDI pitch for Pianos
MIN_MIDI_PITCH = 0 # 0 is the lowest MIDI pitch
MAX_MIDI_PITCH = 127 # 127 is the highest MIDI pitch
MIDI_PITCHES = 128 # 128 is the number of MIDI pitches

MAX_INSTRUMENT_NUMBER = 127

# 0-32: Melody Programs, 32-40: Bass Programs
MEL_PROGRAMS = range(0, 32)  # piano, chromatic percussion, organ, guitar (Melody Programs)
BASS_PROGRAMS = range(32, 40) # bass (Bass Programs)
ELECTRIC_BASS_PROGRAM = 33

# 9 classes: kick, snare, closed_hh, open_hh, low_tom, mid_tom, hi_tom, crash,
# ride
REDUCED_DRUM_PITCH_CLASSES = drums_encoder_decoder.DEFAULT_DRUM_TYPE_PITCHES
# 61 classes: full General MIDI set
FULL_DRUM_PITCH_CLASSES = [
    [p] for p in  # pylint:disable=g-complex-comprehension
    [36, 35, 38, 27, 28, 31, 32, 33, 34, 37, 39, 40, 56, 65, 66, 75, 85, 42, 44,
     54, 68, 69, 70, 71, 73, 78, 80, 46, 67, 72, 74, 79, 81, 45, 29, 41, 61, 64,
     84, 48, 47, 60, 63, 77, 86, 87, 50, 30, 43, 62, 76, 83, 49, 55, 57, 58, 51,
     52, 53, 59, 82]
]
ROLAND_DRUM_PITCH_CLASSES = [
    # kick drum
    [36],
    # snare drum
    [38, 37, 40],
    # closed hi-hat
    [42, 22, 44],
    # open hi-hat
    [46, 26],
    # low tom
    [43, 58],
    # mid tom
    [47, 45],
    # high tom
    [50, 48],
    # crash cymbal
    [49, 52, 55, 57],
    # ride cymbal
    [51, 53, 59]
]

OUTPUT_VELOCITY = 80 # fixed for every note?

CHORD_SYMBOL = note_seq.NoteSequence.TextAnnotation.CHORD_SYMBOL


def _maybe_pad_seqs(seqs, dtype, depth):
  """Pads sequences to match the longest and returns as a numpy array."""
  if not len(seqs):  # pylint:disable=g-explicit-length-test,len-as-condition
    return np.zeros((0, 0, depth), dtype)
  lengths = [len(s) for s in seqs]
  if len(set(lengths)) == 1:
    return np.array(seqs, dtype)
  else:
    length = max(lengths)
    return (np.array([np.pad(s, [(0, length - len(s)), (0, 0)], mode='constant')
                      for s in seqs], dtype))


def _extract_instrument(note_sequence, instrument):
  extracted_ns = copy.copy(note_sequence)
  del extracted_ns.notes[:]
  extracted_ns.notes.extend(
      n for n in note_sequence.notes if n.instrument == instrument)
  return extracted_ns


def maybe_sample_items(seq, sample_size, randomize):
  """Samples a seq if `sample_size` is provided and less than seq size.
  
  """
  if not sample_size or len(seq) <= sample_size:
    return seq
  if randomize:
    indices = set(np.random.choice(len(seq), size=sample_size, replace=False))
    return [seq[i] for i in indices]
  else:
    return seq[:sample_size]


def combine_converter_tensors(converter_tensors, max_num_tensors=None,
                              randomize_sample=True):
  """Combines multiple `ConverterTensors` into one and samples if required."""
  results = []
  for result in converter_tensors:
    results.extend(zip(*result))
  sampled_results = maybe_sample_items(results, max_num_tensors,
                                       randomize_sample)
  if sampled_results:
    return ConverterTensors(*zip(*sampled_results))
  else:
    return ConverterTensors()


def np_onehot(indices, depth, dtype=bool):
  """Converts 1D array of indices to a one-hot 2D array with given depth."""
  #onehot_seq = np.zeros((len(indices), depth), dtype=dtype)
  #onehot_seq[np.arange(len(indices)), indices] = 1.0
  #return onehot_seq
  return np.expand_dims(np.array(indices), 1)



class ConverterTensors(collections.namedtuple(
    'ConverterTensors', ['inputs', 'outputs', 'controls', 'lengths'])):
  """Tuple of tensors output by `to_tensors` method in converters.

  Attributes:
    inputs: Input tensors to feed to the encoder.
    outputs: Output tensors to feed to the decoder.
    controls: (Optional) tensors to use as controls for both encoding and
        decoding.
    lengths: Length of each input/output/control sequence.
  """

  def __new__(cls, inputs=None, outputs=None, controls=None, lengths=None):
    if inputs is None:
      inputs = []
    if outputs is None:
      outputs = []
    if lengths is None:
      lengths = [len(i) for i in inputs]
    if not controls:
      controls = [np.zeros([l, 0]) for l in lengths]
    return super(ConverterTensors, cls).__new__(
        cls, inputs, outputs, controls, lengths)


class BaseNoteSequenceConverter(object):
  """Base class for data converters between items and tensors.

  Inheriting classes must implement the following abstract methods:
    -`to_tensors`
    -`from_tensors`
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self,
               input_depth,
               input_dtype,
               output_depth,
               output_dtype,
               control_depth=0,
               control_dtype=bool,
               end_token=None,
               max_tensors_per_notesequence=None,
               length_shape=(),
               presplit_on_time_changes=True):
    """Initializes BaseNoteSequenceConverter.

    Args:
      input_depth: Depth of final dimension of input (encoder) tensors.
      input_dtype: DType of input (encoder) tensors.
      output_depth: Depth of final dimension of output (decoder) tensors.
      output_dtype: DType of output (decoder) tensors.
      control_depth: Depth of final dimension of control tensors, or zero if not
          conditioning on control tensors.
      control_dtype: DType of control tensors.
      end_token: Optional end token.
      max_tensors_per_notesequence: The maximum number of outputs to return for
          each input.
      length_shape: Shape of length returned by `to_tensor`.
      presplit_on_time_changes: Whether to split NoteSequence on time changes
        before converting.
    """
    self._input_depth = input_depth
    self._input_dtype = input_dtype
    self._output_depth = output_depth
    self._output_dtype = output_dtype
    self._control_depth = control_depth
    self._control_dtype = control_dtype
    self._end_token = end_token
    self._max_tensors_per_input = max_tensors_per_notesequence
    self._str_to_item_fn = note_seq.NoteSequence.FromString
    self._mode = None
    self._length_shape = length_shape
    self._presplit_on_time_changes = presplit_on_time_changes

  def set_mode(self, mode):
    if mode not in ['train', 'eval', 'infer']:
      raise ValueError('Invalid mode: %s' % mode)
    self._mode = mode

  @property
  def is_training(self):
    return self._mode == 'train'

  @property
  def is_inferring(self):
    return self._mode == 'infer'

  @property
  def str_to_item_fn(self):
    return self._str_to_item_fn

  @property
  def max_tensors_per_notesequence(self):
    return self._max_tensors_per_input

  @max_tensors_per_notesequence.setter
  def max_tensors_per_notesequence(self, value):
    self._max_tensors_per_input = value

  @property
  def end_token(self):
    """End token, or None."""
    return self._end_token

  @property
  def input_depth(self):
    """Dimension of inputs (to encoder) at each timestep of the sequence."""
    return self._input_depth

  @property
  def input_dtype(self):
    """DType of inputs (to encoder)."""
    return self._input_dtype

  @property
  def output_depth(self):
    """Dimension of outputs (from decoder) at each timestep of the sequence."""
    return self._output_depth

  @property
  def output_dtype(self):
    """DType of outputs (from decoder)."""
    return self._output_dtype

  @property
  def control_depth(self):
    """Dimension of control inputs at each timestep of the sequence."""
    return self._control_depth

  @property
  def control_dtype(self):
    """DType of control inputs."""
    return self._control_dtype

  @property
  def length_shape(self):
    """Shape of length returned by `to_tensor`."""
    return self._length_shape

  def _sanitize_time_signatures(self, ns):
    """Normalize NoteSequence time signatures to a single 4/4 at time 0.

    Many POP909 MIDIs include multiple time signatures at t=0 (e.g., 4/4 then
    1/4) due to encoding artifacts. This method removes all time signature
    changes and enforces a single 4/4 at the beginning to ensure integer
    steps-per-bar during quantization.
    """
    ts_list = list(ns.time_signatures)
    del ns.time_signatures[:]
    chosen = None
    for ts in ts_list:
      if ts.time <= 1e-3 and ts.numerator == 4 and ts.denominator == 4:
        chosen = ts
        break
    if chosen is None:
      # Create canonical 4/4 at t=0
      chosen = ns.time_signatures.add()
      chosen.time = 0.0
      chosen.numerator = 4
      chosen.denominator = 4
    else:
      # Re-add the chosen 4/4, normalized to t=0
      new_ts = ns.time_signatures.add()
      new_ts.time = 0.0
      new_ts.numerator = 4
      new_ts.denominator = 4
    # Drop any subsequent time signature changes to keep a fixed bar size.

  def _sanitize_tempos(self, ns):
    """Normalize NoteSequence tempos to a single event at time 0.

    Quantization in note_seq requires a constant tempo when using
    `quantize_note_sequence` with fixed `steps_per_quarter`. POP909 often
    includes slight tempo jitter curves. We select the tempo at t≈0 (or the
    earliest tempo) and enforce it globally by removing subsequent changes.

    Returns:
      The selected tempo in QPM (quarters per minute), or 120.0 if no tempos found.
    """
    tempos = list(ns.tempos)
    del ns.tempos[:]
    chosen = None
    for t in tempos:
      if t.time <= 1e-3:
        chosen = t
        break
    if chosen is None and tempos:
      # Pick earliest tempo event
      chosen = min(tempos, key=lambda x: x.time)
    qpm = chosen.qpm if chosen is not None else 120.0
    tempo = ns.tempos.add()
    tempo.time = 0.0
    tempo.qpm = qpm
    return qpm

  @abc.abstractmethod
  def to_tensors(self, item):
    """Python method that converts `item` into list of `ConverterTensors`."""
    pass

  @abc.abstractmethod
  def from_tensors(self, samples, controls=None):
    """Python method that decodes model samples into list of items."""
    pass


class LegacyEventListOneHotConverter(BaseNoteSequenceConverter):
  """Converts NoteSequences using legacy OneHotEncoding framework.

  Quantizes the sequences, extracts event lists in the requested size range,
  uniquifies, and converts to encoding. Uses the OneHotEncoding's
  output encoding for both the input and output.

  Attributes:
    event_list_fn: A function that returns a new EventSequence.
    event_extractor_fn: A function for extracing events into EventSequences. The
      sole input should be the quantized NoteSequence.
    legacy_encoder_decoder: An instantiated OneHotEncoding object to use.
    add_end_token: Whether or not to add an end token. Recommended to be False
      for fixed-length outputs.
    slice_bars: Optional size of window to slide over raw event lists after
      extraction.
    steps_per_quarter: The number of quantization steps per quarter note.
      Mututally exclusive with `steps_per_second`.
    steps_per_second: The number of quantization steps per second.
      Mututally exclusive with `steps_per_quarter`.
    quarters_per_bar: The number of quarter notes per bar.
    pad_to_total_time: Pads each input/output tensor to the total time of the
      NoteSequence.
    max_tensors_per_notesequence: The maximum number of outputs to return
      for each NoteSequence.
    presplit_on_time_changes: Whether to split NoteSequence on time changes
      before converting.
    chord_encoding: An instantiated OneHotEncoding object to use for encoding
      chords on which to condition, or None if not conditioning on chords.
    condition_on_key: If True, condition on key; key is represented as a
      depth-12 one-hot encoding.
    dedupe_event_lists: If True, only keep unique events in the extracted
      event list.
  """

  def __init__(self, event_list_fn, event_extractor_fn,
               legacy_encoder_decoder, add_end_token=False, slice_bars=None,
               slice_steps=None, steps_per_quarter=None, steps_per_second=None,
               quarters_per_bar=4, pad_to_total_time=False,
               max_tensors_per_notesequence=None,
               presplit_on_time_changes=True, chord_encoding=None,
               condition_on_key=False, dedupe_event_lists=True):
    if (steps_per_quarter, steps_per_second).count(None) != 1:
      raise ValueError(
          'Exactly one of `steps_per_quarter` and `steps_per_second` should be '
          'provided.')
    if (slice_bars, slice_steps).count(None) == 0:
      raise ValueError(
          'At most one of `slice_bars` and `slice_steps` should be provided.')
    self._event_list_fn = event_list_fn
    self._event_extractor_fn = event_extractor_fn
    self._legacy_encoder_decoder = legacy_encoder_decoder
    self._chord_encoding = chord_encoding
    self._condition_on_key = condition_on_key
    self._steps_per_quarter = steps_per_quarter
    if steps_per_quarter:
      self._steps_per_bar = steps_per_quarter * quarters_per_bar
    self._steps_per_second = steps_per_second
    if slice_bars:
      self._slice_steps = self._steps_per_bar * slice_bars
    else:
      self._slice_steps = slice_steps
    self._pad_to_total_time = pad_to_total_time
    self._dedupe_event_lists = dedupe_event_lists

    depth = legacy_encoder_decoder.num_classes + add_end_token
    control_depth = (
        chord_encoding.num_classes if chord_encoding is not None else 0
    ) + (
        12 if condition_on_key else 0
    )
    super(LegacyEventListOneHotConverter, self).__init__(
        input_depth=depth,
        input_dtype=bool,
        output_depth=depth,
        output_dtype=bool,
        control_depth=control_depth,
        control_dtype=bool,
        end_token=legacy_encoder_decoder.num_classes if add_end_token else None,
        presplit_on_time_changes=presplit_on_time_changes,
        max_tensors_per_notesequence=max_tensors_per_notesequence)

  def _dedupe_and_sample(self, all_sliced_lists):
    """Dedupe lists of events, chords, and keys, then optionally sample."""
    # This function expects the following lists (when chords & keys are present;
    # when absent those lists will simply be missing):
    #
    # all_sliced_lists = [
    #   [
    #     [event_00, event_01, ..., event_0a],
    #     [event_10, event_11, ..., event_1b],
    #     [event_20, event_21, ..., event_2c]
    #   ],
    #   [
    #     [chord_00, chord_01, ..., chord_0a],
    #     [chord_10, chord_11, ..., chord_1b],
    #     [chord_20, chord_21, ..., chord_2c]
    #   ],
    #   [
    #     [key_00, key_01, ..., key_0a],
    #     [key_10, key_11, ..., key_1b],
    #     [key_20, key_21, ..., key_2c]
    #   ]
    # ]

    sliced_multievent_lists = [zip(*lists) for lists in zip(*all_sliced_lists)]

    # Now we have:
    #
    # sliced_multievent_lists = [
    #   [(event_00, chord_00, key_00), ..., (event_0a, chord_0a, key_0a)],
    #   [(event_10, chord_10, key_10), ..., (event_1b, chord_1b, key_1b)],
    #   [(event_20, chord_20, key_20), ..., (event_2c, chord_2c, key_2c)]
    # ]

    # TODO(adarob): Consider handling the fact that different event lists can
    # be mapped to identical tensors by the encoder_decoder (e.g., Drums).

    if self._dedupe_event_lists:
      multievent_tuples = list(
          set(tuple(l) for l in sliced_multievent_lists))
    else:
      multievent_tuples = [tuple(l) for l in sliced_multievent_lists]
    multievent_tuples = maybe_sample_items(
        multievent_tuples,
        self.max_tensors_per_notesequence,
        self.is_training)

    # Now multievent_tuples is structured like sliced_multievent_lists
    # above, with duplicates optionally removed and sampled.

    if multievent_tuples:
      # Return lists structured like input all_sliced_lists.
      return list(zip(*[zip(*t) for t in multievent_tuples if t]))
    else:
      return []

  def _chords_and_keys_to_controls(self, chord_tuples, key_tuples):
    """Map chord and/or key tuples to control tensors."""
    control_seqs = []

    # Use zip_longest here because chord_tuples and key_tuples should either be:
    #   a) a list of tuples, with chord_tuples and key_tuples the same shape
    #   b) the empty list
    for ct, kt in itertools.zip_longest(chord_tuples, key_tuples):
      controls = []

      if ct is not None:
        try:
          chord_tokens = [self._chord_encoding.encode_event(e) for e in ct]
          if self.end_token:
            # Repeat the last chord instead of using a special token;
            # otherwise the model may learn to rely on the special token to
            # detect endings.
            if chord_tokens:
              chord_tokens.append(chord_tokens[-1])
            else:
              chord_tokens.append(
                  self._chord_encoding.encode_event(note_seq.NO_CHORD))
        except (note_seq.ChordSymbolError, note_seq.ChordEncodingError):
          return []
        controls.append(np_onehot(
            chord_tokens, self._chord_encoding.num_classes,
            self.control_dtype))

      if kt is not None:
        if self.end_token:
          # Repeat the last key. If the sequence is empty, just pick randomly.
          if kt:
            kt.append(kt[-1])
          else:
            kt.append(np.random.choice(range(12)))
        controls.append(np_onehot(kt, 12, self.control_dtype))

      # Concatenate controls (chord and/or key) depthwise. The resulting control
      # tensor should be one of:
      #   a) a one-hot-encoded chord (if not conditioning on key)
      #   b) a one-hot-encoded key (if not conditioning on chord)
      #   c) both (a) and (b), concatenated depthwise
      control_seqs.append(np.concatenate(controls, axis=1))

    return control_seqs

  def to_tensors(self, item):
    """Converts NoteSequence to unique, one-hot tensor sequences."""
    note_sequence = item
    try:
      if self._steps_per_quarter:
        quantized_sequence = note_seq.quantize_note_sequence(
            note_sequence, self._steps_per_quarter)
        if (note_seq.steps_per_bar_in_quantized_sequence(quantized_sequence) !=
            self._steps_per_bar):
          return ConverterTensors()
      else:
        quantized_sequence = note_seq.quantize_note_sequence_absolute(
            note_sequence, self._steps_per_second)
    except (note_seq.BadTimeSignatureError, note_seq.NonIntegerStepsPerBarError,
            note_seq.NegativeTimeError):
      return ConverterTensors()

    if (self._chord_encoding and not any(
        ta.annotation_type == CHORD_SYMBOL
        for ta in quantized_sequence.text_annotations)) or (
            self._condition_on_key and not quantized_sequence.key_signatures):
      # We are conditioning on chords and/or key but sequence does not have
      # them. Try to infer chords and optionally key.
      try:
        note_seq.infer_chords_for_sequence(
            quantized_sequence, add_key_signatures=self._condition_on_key)
      except note_seq.ChordInferenceError:
        return ConverterTensors()

    event_lists, unused_stats = self._event_extractor_fn(quantized_sequence)
    if self._pad_to_total_time:
      for e in event_lists:
        e.set_length(len(e) + e.start_step, from_left=True)
        e.set_length(quantized_sequence.total_quantized_steps)
    if self._slice_steps:
      sliced_event_lists = []
      for l in event_lists:
        for i in range(self._slice_steps, len(l) + 1, self._steps_per_bar):
          sliced_event_lists.append(l[i - self._slice_steps: i])
    else:
      sliced_event_lists = event_lists

    # We are going to dedupe the event lists. However, when conditioning on
    # chords and/or key, we want to include the same event list multiple times
    # if it appears with different chords or keys.
    all_sliced_lists = [sliced_event_lists]

    if self._chord_encoding:
      # Extract chord lists that correspond to event lists, i.e. for each event
      # we find the chord active at that time step.
      try:
        sliced_chord_lists = chords_lib.event_list_chords(
            quantized_sequence, sliced_event_lists)
      except chords_lib.CoincidentChordsError:
        return ConverterTensors()
      all_sliced_lists.append(sliced_chord_lists)

    if self._condition_on_key:
      # Extract key lists that correspond to event lists, i.e. for each event
      # we find the key active at that time step.
      if self._steps_per_second:
        steps_per_second = self._steps_per_second
      else:
        qpm = quantized_sequence.tempos[0].qpm
        steps_per_second = self._steps_per_quarter * qpm / 60.0
      sliced_key_lists = chords_lib.event_list_keys(
          quantized_sequence, sliced_event_lists, steps_per_second)
      all_sliced_lists.append(sliced_key_lists)

    all_unique_tuples = self._dedupe_and_sample(all_sliced_lists)
    if not all_unique_tuples:
      return ConverterTensors()

    unique_event_tuples = all_unique_tuples[0]
    unique_chord_tuples = all_unique_tuples[1] if self._chord_encoding else []
    unique_key_tuples = all_unique_tuples[-1] if self._condition_on_key else []

    if self._chord_encoding or self._condition_on_key:
      # We need to encode control sequences consisting of chords and/or keys.
      control_seqs = self._chords_and_keys_to_controls(
          unique_chord_tuples, unique_key_tuples)
      if not control_seqs:
        return ConverterTensors()
    else:
      control_seqs = []

    seqs = []
    for t in unique_event_tuples:
      seqs.append(np_onehot(
          [self._legacy_encoder_decoder.encode_event(e) for e in t] +
          ([] if self.end_token is None else [self.end_token]),
          self.output_depth, self.output_dtype))

    return ConverterTensors(inputs=seqs, outputs=seqs, controls=control_seqs)

  def from_tensors(self, samples, controls=None):
    """Converts model samples to a list of `NoteSequence`s."""
    output_sequences = []
    for i, sample in enumerate(samples):
      s = sample  # np.argmax(sample, axis=-1)
      if self.end_token is not None and self.end_token in s.tolist():
        end_index = s.tolist().index(self.end_token)
      else:
        end_index = len(s)
      s = s[:end_index]
      event_list = self._event_list_fn()
      for e in s:
        assert e != self.end_token
        event_list.append(self._legacy_encoder_decoder.decode_event(e))
      if self._steps_per_quarter:
        qpm = note_seq.DEFAULT_QUARTERS_PER_MINUTE
        seconds_per_step = 60.0 / (self._steps_per_quarter * qpm)
        sequence = event_list.to_sequence(velocity=OUTPUT_VELOCITY, qpm=qpm)
      else:
        seconds_per_step = 1.0 / self._steps_per_second
        sequence = event_list.to_sequence(velocity=OUTPUT_VELOCITY)
      if self._chord_encoding and controls is not None:
        chords = [self._chord_encoding.decode_event(e)
                  for e in np.argmax(controls[i][:, :-12], axis=-1)[:end_index]]
        chord_times = [step * seconds_per_step for step in event_list.steps]
        chords_lib.add_chords_to_sequence(sequence, chords, chord_times)
      if self._condition_on_key and controls is not None:
        keys = np.argmax(controls[i][:, -12:], axis=-1)[:end_index]
        key_times = [step * seconds_per_step for step in event_list.steps]
        chords_lib.add_keys_to_sequence(sequence, keys, key_times)
      output_sequences.append(sequence)
    return output_sequences


class OneHotMelodyConverter(LegacyEventListOneHotConverter):
  """Converter for legacy MelodyOneHotEncoding.

  Attributes:
    melody_fn: A function that takes no arguments and returns an empty Melody.
    melody_encoding: The MelodyOneHotEncoding object used to encode/decode
        individual melody events.
  """

  def __init__(self, min_pitch=PIANO_MIN_MIDI_PITCH,
               max_pitch=PIANO_MAX_MIDI_PITCH, valid_programs=None,
               skip_polyphony=False, max_bars=None, slice_bars=None,
               gap_bars=1.0, steps_per_quarter=4, quarters_per_bar=4,
               add_end_token=False, pad_to_total_time=False,
               max_tensors_per_notesequence=5, presplit_on_time_changes=True,
               chord_encoding=None, condition_on_key=False,
               dedupe_event_lists=True, strict_tempo=False, instrument=None):
    """Initialize a OneHotMelodyConverter object.

    Args:
      min_pitch: The minimum pitch to model. Those below this value will be
          ignored.
      max_pitch: The maximum pitch to model. Those above this value will be
          ignored.
      valid_programs: Optional set of program numbers to allow.
      skip_polyphony: Whether to skip polyphonic instruments. If False, the
          highest pitch will be taken in polyphonic sections.
      max_bars: Optional maximum number of bars per extracted melody, before
          slicing.
      slice_bars: Optional size of window to slide over raw Melodies after
          extraction.
      gap_bars: If this many bars or more of non-events follow a note event, the
          melody is ended. Disabled when set to 0 or None.
      steps_per_quarter: The number of quantization steps per quarter note.
      quarters_per_bar: The number of quarter notes per bar.
      add_end_token: Whether to add an end token at the end of each sequence.
      pad_to_total_time: Pads each input/output tensor to the total time of the
          NoteSequence.
      max_tensors_per_notesequence: The maximum number of outputs to return
          for each NoteSequence.
      presplit_on_time_changes: Whether to split NoteSequence on time changes
          before converting.
      chord_encoding: An instantiated OneHotEncoding object to use for encoding
          chords on which to condition, or None if not conditioning on chords.
      condition_on_key: If True, condition on key; key is represented as a
          depth-12 one-hot encoding.
      dedupe_event_lists: If True, only keep unique events in the extracted
          event list.
      strict_tempo: If False, sanitize tempo/time sig; if True, keep original.
      instrument: Optional instrument number to filter to. If None, process all
          instruments in the sequence. If set (e.g., 0), only extract notes from
          that instrument.
    """
    self._min_pitch = min_pitch
    self._max_pitch = max_pitch
    self._valid_programs = valid_programs
    self._strict_tempo = strict_tempo
    self._instrument = instrument
    self.last_chosen_tempo = None
    
    steps_per_bar = steps_per_quarter * quarters_per_bar
    max_steps_truncate = steps_per_bar * max_bars if max_bars else None

    def melody_fn():
      return note_seq.Melody(
          steps_per_bar=steps_per_bar, steps_per_quarter=steps_per_quarter)

    self._melody_fn = melody_fn
    self._melody_encoding = note_seq.MelodyOneHotEncoding(
        min_pitch, max_pitch + 1)

    melody_extractor_fn = functools.partial(
        melody_pipelines.extract_melodies,
        min_bars=1,
        gap_bars=gap_bars or float('inf'),
        max_steps_truncate=max_steps_truncate,
        min_unique_pitches=1,
        ignore_polyphonic_notes=not skip_polyphony,
        pad_end=True)
    super(OneHotMelodyConverter, self).__init__(
        melody_fn,
        melody_extractor_fn,
        self._melody_encoding,
        add_end_token=add_end_token,
        slice_bars=slice_bars,
        pad_to_total_time=pad_to_total_time,
        steps_per_quarter=steps_per_quarter,
        quarters_per_bar=quarters_per_bar,
        max_tensors_per_notesequence=max_tensors_per_notesequence,
        presplit_on_time_changes=presplit_on_time_changes,
        chord_encoding=chord_encoding,
        condition_on_key=condition_on_key,
        dedupe_event_lists=dedupe_event_lists)

  @property
  def melody_fn(self):
    return self._melody_fn

  @property
  def melody_encoding(self):
    return self._melody_encoding

  def _to_tensors_fn(self, note_sequence):
    # Sanitize time signatures and tempos before processing
    try:
      self._sanitize_time_signatures(note_sequence)
    except Exception:
      pass
    if not self._strict_tempo:
      try:
        self.last_chosen_tempo = self._sanitize_tempos(note_sequence)
      except Exception:
        self.last_chosen_tempo = None
    
    # Filter to specific instrument if requested
    if self._instrument is not None:
      before_count = len(note_sequence.notes)
      notes = [n for n in note_sequence.notes if n.instrument == self._instrument]
      del note_sequence.notes[:]
      note_sequence.notes.extend(notes)
      if before_count > 0 and len(notes) == 0:
        print(f"Warning: Filtered instrument {self._instrument}, had {before_count} notes, now have 0")
    
    def is_valid(note):
      if (self._valid_programs is not None and
          note.program not in self._valid_programs):
        return False
      return self._min_pitch <= note.pitch <= self._max_pitch
    notes = list(note_sequence.notes)
    del note_sequence.notes[:]
    note_sequence.notes.extend([n for n in notes if is_valid(n)])
    return super(OneHotMelodyConverter, self).to_tensors(note_sequence)

  def to_tensors(self, item):
    note_sequence = item
    return split_process_and_combine(note_sequence,
                                     self._presplit_on_time_changes,
                                     self.max_tensors_per_notesequence,
                                     self.is_training, self._to_tensors_fn)


def split_process_and_combine(note_sequence, split, sample_size, randomize,
                              to_tensors_fn):
  """Splits a `NoteSequence`, processes and combines the `ConverterTensors`.

  Args:
    note_sequence: The `NoteSequence` to split, process and combine.
    split: If True, the given note_sequence is split into multiple based on time
      changes, and the tensor outputs are concatenated.
    sample_size: Outputs are sampled if size exceeds this value.
    randomize: If True, outputs are randomly sampled (this is generally done
      during training).
    to_tensors_fn: A fn that converts a `NoteSequence` to `ConverterTensors`.

  Returns:
    A `ConverterTensors` obj.
  """
  note_sequences = sequences_lib.split_note_sequence_on_time_changes(
      note_sequence) if split else [note_sequence]
  results = []
  for ns in note_sequences:
    tensors = to_tensors_fn(ns)
    sampled_results = maybe_sample_items(
        list(zip(*tensors)), sample_size, randomize)
    if sampled_results:
      results.append(ConverterTensors(*zip(*sampled_results)))
    else:
      results.append(ConverterTensors())
  return combine_converter_tensors(results, sample_size, randomize)


class POP909TrioConverter(BaseNoteSequenceConverter):
  """Converter for POP909 3-track (MELODY, BRIDGE, PIANO) arrangements.

  Extracts tracks 0 (MELODY), 1 (BRIDGE), and 2 (PIANO) from a NoteSequence
  and concatenates their one-hot melody encodings along the feature axis.

  Attributes:
    slice_bars: Optional size of window to slide over full converted tensor.
    max_bars: Optional maximum number of bars per extracted sequence, before
      slicing.
    steps_per_quarter: The number of quantization steps per quarter note.
    quarters_per_bar: The number of quarter notes per bar.
    max_tensors_per_notesequence: The maximum number of outputs to return
      for each NoteSequence.
  """

  def __init__(self, slice_bars=None, max_bars=1024, steps_per_quarter=4,
               quarters_per_bar=4, max_tensors_per_notesequence=5,
               presplit_on_time_changes=True, gap_bars=None, strict_tempo=False):
    """Initialize a POP909TrioConverter.

    Args:
      slice_bars: Optional window size in bars to slide over full tensor.
      max_bars: Maximum bars per extracted sequence before slicing.
      steps_per_quarter: Quantization steps per quarter note.
      quarters_per_bar: Quarter notes per bar.
      max_tensors_per_notesequence: Max outputs per NoteSequence.
      presplit_on_time_changes: Whether to split on time changes before converting.
      gap_bars: Ignored; present for API compatibility with OneHotMelodyConverter.
    """
    self._slice_bars = slice_bars
    self._max_bars = max_bars
    self._steps_per_quarter = steps_per_quarter
    self._steps_per_bar = steps_per_quarter * quarters_per_bar
    self._strict_tempo = strict_tempo
    self.last_chosen_tempo = None

    # Create a converter for each of the three tracks (melody-like encoding).
    # Don't filter by instrument since we extract each track separately
    self._track_converter = OneHotMelodyConverter(
        gap_bars=None,
        steps_per_quarter=steps_per_quarter,
        quarters_per_bar=quarters_per_bar,
        pad_to_total_time=True,
        presplit_on_time_changes=False,
        max_tensors_per_notesequence=None,
        strict_tempo=strict_tempo,
        instrument=None)

    # Output depth is 3x the melody encoding depth (concatenated).
    output_depth = self._track_converter.output_depth * 3

    super(POP909TrioConverter, self).__init__(
        input_depth=output_depth,
        input_dtype=bool,
        output_depth=output_depth,
        output_dtype=bool,
        control_depth=0,
        control_dtype=bool,
        end_token=False,
        presplit_on_time_changes=presplit_on_time_changes,
        max_tensors_per_notesequence=max_tensors_per_notesequence)

  def _to_tensors_fn(self, note_sequence):
    """Converts a `NoteSequence` to `ConverterTensors` by concatenating 3 tracks."""
    # Sanitize spurious time signature changes at t=0 (e.g., 4/4 -> 1/4).
    # POP909 arrangements are intended to be in 4/4. Some MIDIs carry
    # artifacts declaring 1/4 or 2/4 at t=0 which break quantization.
    # We normalize to a single 4/4 time signature at t=0.
    try:
      self._sanitize_time_signatures(note_sequence)
    except Exception:
      # If sanitation fails, continue; quantization may still succeed.
      pass
    # Enforce a single tempo at t=0 to avoid MultipleTempo errors during
    # quantization. POP909 provides detailed tempo curves; for fixed-grid
    # quantization we clamp to the first tempo.
    if not self._strict_tempo:
      try:
        self.last_chosen_tempo = self._sanitize_tempos(note_sequence)
      except Exception:
        self.last_chosen_tempo = None
    try:
      quantized_sequence = note_seq.quantize_note_sequence(
          note_sequence, self._steps_per_quarter)
      if (note_seq.steps_per_bar_in_quantized_sequence(quantized_sequence) !=
          self._steps_per_bar):
        return ConverterTensors()
    except (note_seq.BadTimeSignatureError, note_seq.NonIntegerStepsPerBarError,
            note_seq.NegativeTimeError):
      return ConverterTensors()

    # Set total time for consistent padding.
    total_bars = int(
        np.ceil(quantized_sequence.total_quantized_steps / self._steps_per_bar))
    total_bars = min(total_bars, self._max_bars)
    note_sequence.total_time = (
        total_bars * self._steps_per_bar *
        60 / note_sequence.tempos[0].qpm / self._steps_per_quarter)

    # Extract the three tracks (0=MELODY, 1=BRIDGE, 2=PIANO).
    track_tensors = []
    for track_idx in range(3):
      # Create a NoteSequence with only notes from this track.
      track_ns = _extract_instrument(note_sequence, track_idx)
      tensors = self._track_converter.to_tensors(track_ns)
      if tensors.outputs:
        track_tensors.append(tensors.outputs[0])
      else:
        # If a track is empty, return no sequences (trio requires all 3).
        return ConverterTensors()

    # Concatenate the three tracks along the feature axis.
    seqs = []
    if len(track_tensors) == 3 and all(len(t) == len(track_tensors[0]) for t in track_tensors):
      combined = np.concatenate(track_tensors, axis=-1)
      
      # Optionally slice the combined tensor.
      if self._slice_bars:
        slice_steps = self._slice_bars * self._steps_per_bar
        for j in range(0, len(combined) - slice_steps + 1, self._steps_per_bar):
          seqs.append(combined[j:j + slice_steps])
      else:
        seqs.append(combined)

    return ConverterTensors(inputs=seqs, outputs=seqs)

  def to_tensors(self, item):
    note_sequence = item
    return split_process_and_combine(note_sequence,
                                     self._presplit_on_time_changes,
                                     self.max_tensors_per_notesequence,
                                     self.is_training, self._to_tensors_fn)

  def from_tensors(self, samples, controls=None):
    """
    Decodes Trio indices (Column 0=Melody, 1=Bridge, 2=Piano) back to NoteSequence.
    """
    output_sequences = []
    
    for s in samples:
      # s shape is (Length, 3)
      
      # 1. Extract Tracks (Flattening to 1D arrays)
      # We use s[:, 0] instead of s[:, 0:1] to ensure we get shape (L,) not (L, 1)
      mel_tensor = s[:, 0] 
      bridge_tensor = s[:, 1]
      piano_tensor = s[:, 2]

      # 2. Decode each track separately using the sub-converter
      # Note: We wrap them in a list [] because from_tensors expects a batch of samples
      mel_ns = self._track_converter.from_tensors([mel_tensor])[0]
      bridge_ns = self._track_converter.from_tensors([bridge_tensor])[0]
      piano_ns = self._track_converter.from_tensors([piano_tensor])[0]

      # 3. Assign Instruments
      # Melody is already 0
      for n in bridge_ns.notes:
        n.instrument = 1
      for n in piano_ns.notes:
        n.instrument = 2

      # 4. Merge
      ns = mel_ns
      ns.notes.extend(bridge_ns.notes)
      ns.notes.extend(piano_ns.notes)
      
      # Recalculate total time
      if ns.notes:
          ns.total_time = max(n.end_time for n in ns.notes)
      
      output_sequences.append(ns)
      
    return output_sequences


# ============================================================================
# OCTUPLE CONVERTERS
# ============================================================================

class POP909OctupleMelodyConverter(BaseNoteSequenceConverter):
  """
  Octuple Melody converter aligned with the new Trio logic.
  1. Uses split_process_and_combine to slice data into bars (e.g. 64).
  2. Extracts Inst 0 (Melody) from the slice.
  3. Encodes it directly (supporting potential polyphony).
  """

  def __init__(self, slice_bars=64, max_tensors_per_notesequence=5, gap_bars=None,
               steps_per_quarter=4, quarters_per_bar=4, strict_tempo=False, presplit_on_time_changes=True):
    self._oct_encoding = OctupleEncoding()
    self._slice_bars = slice_bars
    self._steps_per_quarter = steps_per_quarter
    self._quarters_per_bar = quarters_per_bar
    self._strict_tempo = strict_tempo
    
    super(POP909OctupleMelodyConverter, self).__init__(
        input_depth=1, input_dtype=object,
        output_depth=1, output_dtype=object,
        max_tensors_per_notesequence=max_tensors_per_notesequence,
        presplit_on_time_changes=presplit_on_time_changes)

  def _to_tensors_fn(self, note_sequence):
    # 1. Sanitize & Quantize
    try:
      self._sanitize_time_signatures(note_sequence)
    except Exception:
      pass
    if not self._strict_tempo:
      try:
        self.last_chosen_tempo = self._sanitize_tempos(note_sequence)
      except Exception:
        self.last_chosen_tempo = None
    try:
      quantized_sequence = note_seq.quantize_note_sequence(
          note_sequence, self._steps_per_quarter)
    except Exception:
      return ConverterTensors()

    # 2. Extract and Encode Melody (Track 0)
    inst_ns = _extract_instrument(quantized_sequence, 0)
    if len(inst_ns.notes) == 0:
        return ConverterTensors()

    try:
        full_song = self._oct_encoding.encode_notesequence(inst_ns)
    except Exception:
        return ConverterTensors()

    if len(full_song) == 0:
        return ConverterTensors()
    
    # Set Instrument ID 0
    full_song[:, 2] = 0
    
    # Sort (Primary: Bar (0), Secondary: Position (1))
    # Using stable sort: separate argsorts must be done Secondary first, then Primary.
    full_song = full_song[np.argsort(full_song[:, 1], kind='stable')]
    full_song = full_song[np.argsort(full_song[:, 0], kind='stable')]

    # 3. Deterministic Slicing with Stride
    max_bar = full_song[:, 0].max()
    min_bar = full_song[:, 0].min()
    total_bars = max_bar - min_bar + 1
    
    stride = 32 # 50% overlap

    sliced_outputs = []

    if total_bars <= self._slice_bars:
        slice_data = full_song.copy()
        slice_data[:, 0] -= min_bar
        sliced_outputs.append(slice_data.astype(np.int32))
    else:
        # Standard stride + Tail + Unique
        valid_starts = np.arange(min_bar, max_bar - self._slice_bars + 1, stride)
        last_possible_start = max_bar - self._slice_bars
        valid_starts = np.append(valid_starts, last_possible_start)
        valid_starts = np.unique(valid_starts)

        for start_bar in valid_starts:
            end_bar = start_bar + self._slice_bars
            mask = (full_song[:, 0] >= start_bar) & (full_song[:, 0] < end_bar)
            slice_data = full_song[mask].copy()
            
            if len(slice_data) > 0:
                # CRITICAL: Shift Bar tokens
                slice_data[:, 0] -= start_bar
                sliced_outputs.append(slice_data.astype(np.int32))

    if not sliced_outputs:
        return ConverterTensors()

    return ConverterTensors(inputs=sliced_outputs, outputs=sliced_outputs)
  
  def to_tensors(self, item):
    note_sequence = item
    return split_process_and_combine(note_sequence,
                                     self._presplit_on_time_changes,
                                     self.max_tensors_per_notesequence,
                                     self.is_training, 
                                     self._to_tensors_fn)

  def from_tensors(self, samples, controls=None):
    """
    Converts Octuple token arrays back into NoteSequence objects.
    """
    output_sequences = []
    
    for i, s in enumerate(samples):
        # 1. Decode Numpy Tokens -> MidiToolkit Object using OctupleEncoding
        try:
            midi_obj = self._oct_encoding.decode(s)
        except Exception as e:
            print(f"Error decoding Octuple sample {i}: {e}")
            continue

        # 2. Convert MidiToolkit -> NoteSequence via Temporary File
        # We use a temp file because there is no direct bridge from miditoolkit -> NoteSequence
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
            temp_name = tmp.name
        
        try:
            # Write MIDI to disk
            midi_obj.dump(temp_name)
            
            # Read MIDI back as NoteSequence
            with open(temp_name, 'rb') as f:
                ns = note_seq.midi_to_note_sequence(f.read())
            
            output_sequences.append(ns)
            
        except Exception as e:
            print(f"Error converting MIDI to NoteSequence for sample {i}: {e}")
            output_sequences.append(note_seq.NoteSequence())
            
        finally:
            # Cleanup
            if os.path.exists(temp_name):
                try:
                    os.remove(temp_name)
                except Exception:
                    pass

    return output_sequences

class POP909OctupleTrioConverter(BaseNoteSequenceConverter):
  """
  Octuple Trio converter that aligns with the OneHot logic:
  1. Uses split_process_and_combine to slice data into bars (e.g. 64).
  2. Extracts Inst 0, 1, 2 from the slice.
  3. Encodes them directly (supporting polyphony).
  """

  def __init__(self, slice_bars=64, max_tensors_per_notesequence=5, gap_bars=None,
               steps_per_quarter=4, quarters_per_bar=4, strict_tempo=False, presplit_on_time_changes=True):
    self._oct_encoding = OctupleEncoding()
    self._slice_bars = slice_bars
    self._steps_per_quarter = steps_per_quarter
    self._quarters_per_bar = quarters_per_bar
    self._strict_tempo = strict_tempo
    
    # Octuple uses 8 tokens per event; trio has 3 tracks. 
    # Output depth/dims are handled by the encoding content, not fixed dimensions here.
    super(POP909OctupleTrioConverter, self).__init__(
        input_depth=1, input_dtype=object,
        output_depth=1, output_dtype=object,
        max_tensors_per_notesequence=max_tensors_per_notesequence,
        presplit_on_time_changes=presplit_on_time_changes)

  def _to_tensors_fn(self, note_sequence):
    # 1. Sanitize & Quantize (Whole File)
    try:
      self._sanitize_time_signatures(note_sequence)
    except Exception:
      pass
    if not self._strict_tempo:
      try:
        self.last_chosen_tempo = self._sanitize_tempos(note_sequence)
      except Exception:
        self.last_chosen_tempo = None
    try:
      quantized_sequence = note_seq.quantize_note_sequence(
          note_sequence, self._steps_per_quarter)
    except Exception:
      return ConverterTensors()

    # 2. Encode ALL tracks fully (Melody, Bridge, Piano)
    track_encodings = []
    for inst_idx in range(3):
        inst_ns = _extract_instrument(quantized_sequence, inst_idx)
        
        # If a track is totally empty in the whole file, we can't make a valid Trio
        if len(inst_ns.notes) == 0:
             return ConverterTensors()

        try:
            encoding = self._oct_encoding.encode_notesequence(inst_ns)
        except Exception:
            return ConverterTensors()

        if len(encoding) == 0:
            return ConverterTensors()

        # Set Instrument ID (Column 2)
        encoding[:, 2] = inst_idx
        track_encodings.append(encoding)

    if len(track_encodings) != 3:
        return ConverterTensors()

    # 3. Merge into one giant sorted array
    full_song = np.vstack(track_encodings)
    # Sort by Bar (col 0), then Position (col 1)
    # Using stable sort: Secondary key (Position) first, then Primary key (Bar)
    full_song = full_song[np.argsort(full_song[:, 1], kind='stable')]
    full_song = full_song[np.argsort(full_song[:, 0], kind='stable')]

    # 4. Deterministic Slicing with Stride
    max_bar = full_song[:, 0].max()
    min_bar = full_song[:, 0].min()
    total_bars = max_bar - min_bar + 1
    
    stride = 32  # 50% overlap for 64-bar slices

    sliced_outputs = []

    if total_bars <= self._slice_bars:
        # Short song: Take one slice (0 to end)
        slice_data = full_song.copy()
        slice_data[:, 0] -= min_bar 
        sliced_outputs.append(slice_data.astype(np.int32))
    
    else:
        # Long song: Calculate all valid start points
        # 1. Standard stride
        valid_starts = np.arange(min_bar, max_bar - self._slice_bars + 1, stride)
        
        # 2. "Tail" Logic: Ensure we capture the very end of the song
        last_possible_start = max_bar - self._slice_bars
        valid_starts = np.append(valid_starts, last_possible_start)
        
        # 3. Remove duplicates (in case the stride landed exactly on the tail)
        valid_starts = np.unique(valid_starts)

        # 4. Extract every single valid slice
        for start_bar in valid_starts:
            end_bar = start_bar + self._slice_bars
            
            # Select rows
            mask = (full_song[:, 0] >= start_bar) & (full_song[:, 0] < end_bar)
            slice_data = full_song[mask].copy()
            
            if len(slice_data) > 0:
                # CRITICAL: Shift Bar tokens so slice starts at Bar 0
                slice_data[:, 0] -= start_bar
                sliced_outputs.append(slice_data.astype(np.int32))

    if not sliced_outputs:
        return ConverterTensors()

    # Return ALL slices found (Ignore max_tensors_per_notesequence limit)
    return ConverterTensors(inputs=sliced_outputs, outputs=sliced_outputs)

  def to_tensors(self, item):
    """
    Standard entry point. Uses split_process_and_combine to handle the 
    slicing logic (sliding window) for us.
    """
    note_sequence = item
    return split_process_and_combine(note_sequence,
                                     self._presplit_on_time_changes,
                                     self.max_tensors_per_notesequence,
                                     self.is_training, 
                                     self._to_tensors_fn)

  def from_tensors(self, samples, controls=None):
    """
    Converts Octuple token arrays back into NoteSequence objects (Trio).
    """
    output_sequences = []
    
    for i, s in enumerate(samples):
        try:
            # 1. Decode (Handles Melody, Bridge, Piano reconstruction internally)
            midi_obj = self._oct_encoding.decode(s)
        except Exception as e:
            print(f"Error decoding Octuple Trio sample {i}: {e}")
            continue

        # 2. Convert to NoteSequence
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
            temp_name = tmp.name
        
        try:
            midi_obj.dump(temp_name)
            with open(temp_name, 'rb') as f:
                ns = note_seq.midi_to_note_sequence(f.read())
            output_sequences.append(ns)
        except Exception as e:
            print(f"Error converting Trio MIDI to NoteSequence for sample {i}: {e}")
            output_sequences.append(note_seq.NoteSequence())
        finally:
            if os.path.exists(temp_name):
                try:
                    os.remove(temp_name)
                except Exception:
                    pass

    return output_sequences