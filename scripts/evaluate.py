# Copyright 2022 Klangio GmbH.
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
"""Evaluate Baseline Model for Strumming Action Transcription."""

# system imports
import os
import glob
from typing import Dict, List

# additional imports
import numpy as np
import pandas as pd
import librosa
import mir_eval


def detect_strumming_action(x_audio: np.ndarray, sr: int, x_motion: np.ndarray,
                            t_motion: np.ndarray) -> Dict[str, List[float]]:

  # detect onsets using spectral flux
  x_onset = librosa.onset.onset_strength(y=x_audio, sr=sr)
  onset_times = librosa.onset.onset_detect(onset_envelope=x_onset, sr=sr, units='time')

  # onset pruning within 150ms
  onset_times = onset_times[np.diff(onset_times, prepend=[-1]) > 0.15]

  # smooth motion signal and calculate derivation
  kernel_size = 5
  kernel = np.ones(kernel_size) / kernel_size
  x_motion = np.convolve(x_motion, kernel, mode='same')
  x_motion_gradient = np.gradient(x_motion) * 3

  strums_est = {'U': [], 'D': []}
  for t in onset_times:
    i = np.abs(t_motion - t).argmin()
    if x_motion_gradient[i] > 0:
      strums_est['U'].append(t)
    else:
      strums_est['D'].append(t)

  return strums_est


def main():

  example_counter = 0
  metrics_avg = {}

  for fn_strums in glob.glob('dataset/*.strums'):
    # load audio signal
    fn_audio = fn_strums.replace('.strums', '_line.wav')
    x_audio, sr = librosa.load(fn_audio)

    # load motion signal
    fn_motion = fn_strums.replace('.strums', '.csv')
    df_motion = pd.read_csv(fn_motion, header=None, names=['time', 'value'])
    x_motion = np.array(df_motion['value'])
    t_motion = np.array(df_motion['time'])
    x_motion /= np.max(np.abs(x_motion))

    # load labels
    df_labels = pd.read_csv(fn_strums, sep='\t', header=None, names=['time', 'strum', 'chord'])

    # estimate strums
    strums_est = detect_strumming_action(x_audio, sr, x_motion, t_motion)

    times_est_down = np.sort(np.array(strums_est['D']))
    times_est_up = np.sort(np.array(strums_est['U']))

    times_ref_down = np.sort(df_labels[df_labels['strum'] == 'D']['time'].to_numpy())
    times_ref_up = np.sort(df_labels[df_labels['strum'] == 'U']['time'].to_numpy())

    f1_down, p_down, r_down = mir_eval.onset.f_measure(times_ref_down, times_est_down, window=0.1)
    f1_up, p_up, r_up = mir_eval.onset.f_measure(times_ref_up, times_est_up, window=0.1)

    metrics = {
        'down_f_measure': f1_down,
        'down_precision': p_down,
        'down_recall': r_down,
        'up_f_measure': f1_up,
        'up_precision': p_up,
        'up_recall': r_up,
    }
    metrics_avg = {
        key: metrics_avg.get(key, 0) + metrics.get(key, 0)
        for key in set(metrics_avg) | set(metrics)
    }
    example_counter += 1

    print(f'### Name={os.path.basename(fn_strums)} ###')
    print('[Downstroke] Precision: {:.2%}, Recall: {:.2%}, F-Measure: {:.2%}'.format(
        metrics['down_precision'], metrics['down_recall'], metrics['down_f_measure']))
    print('[Upstroke] Precision: {:.2%}, Recall: {:.2%}, F-Measure: {:.2%}'.format(
        metrics['up_precision'], metrics['up_recall'], metrics['up_f_measure']))
    print()

  # calculate average metrics
  metrics_avg = {key: metrics_avg.get(key, 0) / example_counter for key in set(metrics_avg)}
  print('### Average Results ###')
  print('[Downstroke] Precision: {:.2%}, Recall: {:.2%}, F-Measure: {:.2%}'.format(
      metrics_avg['down_precision'], metrics_avg['down_recall'], metrics_avg['down_f_measure']))
  print('[Upstroke] Precision: {:.2%}, Recall: {:.2%}, F-Measure: {:.2%}'.format(
      metrics_avg['up_precision'], metrics_avg['up_recall'], metrics_avg['up_f_measure']))


if __name__ == '__main__':
  main()