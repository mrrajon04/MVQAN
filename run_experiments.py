#!/usr/bin/env python3
"""MVQAN-CORAL baseline and branch-ablation experiments.

Twelve configurations, three datasets, seeds 11/22/33, one fixed fold.
Run: CUDA_VISIBLE_DEVICES=0 python run_experiments.py
"""
from __future__ import annotations
import os
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
import sys
import importlib.util
import importlib.metadata as package_metadata
from pathlib import Path

MODEL_NAMES = ('LaBraM', 'CBraMod', 'REVE', 'MAtt', 'SPDNet', 'TSMNet_SPDDSMBN', 'SPDIM', 'EA_EEGNet', 'EA_TS_LR', 'Residual_only', 'Residual_only_matched', 'Quantum_only')
DATASET_NAMES = ('BNCI', 'UCL', 'EEGET')
SEED_VALUES = (11, 22, 33)
def parse_arguments(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='MVQAN-CORAL baseline and branch-ablation experiments.')
    parser.add_argument('--data-root', type=Path, default=Path('data'),
                        help='Dataset directory; public data are downloaded when required.')
    parser.add_argument('--output-root', type=Path, default=Path('outputs'),
                        help='Experiment outputs and resumable job artifacts.')
    parser.add_argument('--pretrained-metadata', type=Path,
                        help='Directory containing pretrained_MODEL.json checkpoint identities.')
    parser.add_argument('--plan', action='store_true', help='Print the experiment plan and exit.')
    parser.add_argument('--self-test-only', action='store_true',
                        help='Run numerical checks on CUDA without training experiments.')
    return parser.parse_args(argv)

ARGS = parse_arguments() if __name__ == '__main__' else None
if ARGS is not None and ARGS.plan:
    import json
    print(json.dumps({'datasets': DATASET_NAMES, 'fold_id': '0',
                      'seeds': SEED_VALUES, 'models': MODEL_NAMES,
                      'runs_per_dataset': 36, 'total_runs': 108,
                      'test_units': {'BNCI': 'S01', 'UCL': 'first sorted participant',
                                     'EEGET': 'ALS_01 plus paired held-out control'},
                      'sd': 'sample SD across the three seeds, ddof=1'}, indent=2))
    raise SystemExit(0)


def check_dependencies():
    required = ('numpy', 'scipy', 'pandas', 'scikit-learn', 'torch', 'mne',
                'moabb', 'pyriemann', 'pennylane', 'braindecode', 'joblib',
                'huggingface_hub', 'safetensors')
    missing = []
    for name in required:
        try:
            package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            missing.append(name)
    if missing:
        raise RuntimeError('Install requirements.txt in the active Python environment. Missing: ' + ', '.join(missing))
    for name, version in (('braindecode', '1.5.1'), ('pennylane', '0.45.1')):
        if package_metadata.version(name) != version:
            raise RuntimeError(f'This experiment requires {name}=={version}.')


check_dependencies()
import copy
import contextlib
import gc
import hashlib
import inspect
import itertools
import json
import logging
import math
import random
import shutil
import tempfile
import time
import traceback
import types
import urllib.request
import warnings
import zipfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from fractions import Fraction
import numpy as np
import pandas as pd
from scipy import io as scipy_io, linalg, signal
from scipy import signal as sp_signal
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import mne
import torch
from torch import nn
from torch.nn import functional as F
import pennylane as qml
from pyriemann.estimation import Covariances, XdawnCovariances
from pyriemann.tangentspace import TangentSpace

SOURCE_CODE_SHA256 = 'd99a2ab36bfb1b719c66a0e4e8896b58dd9e28e0832cf68f86bee8c2dac8edc0'
FM_SFREQ = 200
FM_REPOS = {'LaBraM': ('Labram', 'braindecode/labram-pretrained'),
            'CBraMod': ('CBraMod', 'braindecode/cbramod-pretrained'),
            'REVE': ('REVE', 'brain-bzh/reve-base')}
LABRAM_URL = 'https://huggingface.co/braindecode/Labram-Braindecode/resolve/main/braindecode_labram_base.pt'
TRANSDUCTIVE = {'EA_EEGNet', 'EA_TS_LR', 'TSMNet_SPDDSMBN', 'SPDIM'}
FOUNDATION = {'LaBraM', 'CBraMod', 'REVE'}
VIEW_MODELS = {'Quantum_only', 'Residual_only', 'Residual_only_matched'}
DATASET_DISPLAY_NAMES = {'BNCI': 'BNCI2014_009', 'UCL': 'UCL_ALS_MI_LR', 'EEGET': 'EEGET_ALS_CONTROL'}
UCL_CHANNELS = ('Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2')


@dataclass
class Config:
    data_root: str = str(Path.cwd() / 'data')
    output_root: str = str(Path.cwd() / 'outputs')
    ucl_manifest_csv: str = ''
    datasets: tuple = DATASET_NAMES
    device: str = 'cuda:0'
    download_public_data: bool = True
    processed_cache: bool = True
    detailed_console: bool = False
    registry_seed: int = 42
    validation_subjects: int = 1
    fs: int = 128
    p300_tmin: float = 0.0
    p300_tmax: float = 0.8
    ucl_seconds: float = 2.0
    ucl_sfreq: float = 256.0
    # Table 1, Dataset 2 (eight ALS participants), in the authors' paper.
    ucl_sfreq_source: str = 'https://discovery.ucl.ac.uk/id/eprint/10205463/1/Final%20Submission.pdf#Table1'
    ucl_units: str = 'uV'
    eeget_seconds: float = 2.0
    eeget_window_cap: int = 160
    eeget_source_controls: str = 'all_eligible'
    balance_eeget_source: bool = True
    q: int = 6
    depth: int = 2
    fuse_quantum_views: bool = True
    residual_dim: int = 16
    dropout: float = 0.2
    attention_temperature: float = 0.75
    pca_cap: int = 32
    supervised_cap: int = 16
    batch_size: int = 64
    lambda_cor: float = 0.01
    lambda_sep: float = 0.015


@dataclass
class RunConfig:
    datasets: tuple = DATASET_NAMES
    models: tuple = MODEL_NAMES
    seeds: tuple = SEED_VALUES
    epochs: int = 120
    patience: int = 20
    batch_size: int | None = None
    per_group: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    fm_epochs: int = 40
    fm_patience: int = 10
    fm_lr_backbone: float = 1e-4
    fm_lr_head: float = 1e-3
    fm_weight_decay: float = 0.05
    fm_batch_size: int = 32
    fm_micro_batch_size: int = 8
    predict_batch_size: int = 32
    spdim_steps: int = 200
    spdim_lr: float = 1e-2
    matched_tolerance: float = 0.01


CFG = Config()
REB = RunConfig()


def canonical_json(value):

    def convert(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=convert)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


@contextlib.contextmanager
def execution_log(cfg, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', buffering=1) as stream:
        if cfg.detailed_console:
            try:
                yield
            except BaseException:
                traceback.print_exc(file=stream)
                raise
            return
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter('%(name)s %(levelname)s: %(message)s'))
        loggers = [logging.getLogger(name) for name in ('', 'mne', 'moabb', 'pooch')]
        previous = [(logger, logger.handlers[:], logger.propagate) for logger in loggers]
        try:
            for logger in loggers:
                logger.handlers = [handler]
                logger.propagate = False
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream), warnings.catch_warnings():
                warnings.simplefilter('default')
                try:
                    yield
                except BaseException:
                    traceback.print_exc(file=stream)
                    raise
        finally:
            for logger, handlers, propagate in previous:
                logger.handlers = handlers
                logger.propagate = propagate
            handler.close()

@dataclass
class EEGData:
    name: str
    X: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    meta: pd.DataFrame
    fs: float
    channels: tuple
    provenance: dict = field(default_factory=dict)

    def validate(self):
        n = len(self.y)
        if self.X.ndim != 3 or self.X.shape[0] != n or len(self.subject) != n or (len(self.meta) != n):
            raise ValueError(f'{self.name}: inconsistent trial dimensions')
        if self.X.shape[1] != len(self.channels) or not np.isfinite(self.X).all():
            raise ValueError(f'{self.name}: invalid EEG channels or nonfinite samples')
        if set(np.unique(self.y)) != {0, 1} or self.meta.epoch_id.duplicated().any():
            raise ValueError(f'{self.name}: requires binary labels and unique epoch_id')
        if not np.array_equal(self.meta.subject.astype(str).values, self.subject.astype(str)):
            raise ValueError('Participant metadata does not match trials')
        return self


def read_url_json(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'MVQAN-Revision/1.0'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('Archive contains an unsafe path')
        z.extractall(destination)


def download_figshare(article_id, destination, version=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    url = f'https://api.figshare.com/v2/articles/{article_id}'
    if version is not None:
        url += f'/versions/{version}'
    article = read_url_json(url)
    atomic_json(destination / 'figshare_metadata.json', article)
    if not article.get('files'):
        raise ValueError('The public archive has no downloadable files.')
    for record in article['files']:
        name = Path(record['name']).name
        path = destination / name
        expected = record.get('computed_md5') or record.get('supplied_md5')

        def valid_file():
            if not path.exists() or path.stat().st_size != record['size']:
                return False
            if expected:
                h = hashlib.md5()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1 << 20), b''):
                        h.update(block)
                return h.hexdigest() == expected
            return True
        if not valid_file():
            temp = path.with_name(path.name + '.partial')
            request = urllib.request.Request(record['download_url'], headers={'User-Agent': 'EEG-Assignment/2.0'})
            for attempt in range(3):
                try:
                    print(f"Downloading {name}: {record['size'] / 1000000.0:.1f} MB", flush=True)
                    with urllib.request.urlopen(request, timeout=120) as response, temp.open('wb') as stream:
                        shutil.copyfileobj(response, stream, length=1 << 20)
                    os.replace(temp, path)
                    if not valid_file():
                        raise OSError('Size/checksum verification failed: ' + name)
                    break
                except (OSError, TimeoutError):
                    if attempt == 2:
                        raise
        if zipfile.is_zipfile(path):
            destination_folder = destination / path.stem
            stamp = destination / (path.stem + '.extracted.json')
            identity = {'size': path.stat().st_size, 'md5': expected, 'file_id': record['id']}
            if not stamp.exists() or json.loads(stamp.read_text()) != identity or (not destination_folder.exists()):
                safe_extract(path, destination_folder)
                atomic_json(stamp, identity)
    return article


def fixed_bandpass(X, fs, low, high):
    if not 0 < low < high < fs / 2:
        raise ValueError(f'Invalid passband {low}–{high} at sampling rate {fs}')
    sos = signal.butter(4, (low, high), btype='bandpass', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, X, axis=-1).astype(np.float32)


def resample_array(X, old_fs, new_fs):
    from fractions import Fraction
    ratio = Fraction(float(new_fs) / float(old_fs)).limit_denominator(10000)
    return signal.resample_poly(X, ratio.numerator, ratio.denominator, axis=-1).astype(np.float32)


def extract_character_metadata(raw, events, labels, subject, session, run):
    records = []
    sfreq = raw.info['sfreq']
    flash_channels = [s for s in raw.ch_names if s.lower() == 'flash stim']
    if len(flash_channels) != 1:
        return pd.DataFrame({'character_valid': np.zeros(len(events), dtype=bool)})
    flash = raw.get_data(picks=flash_channels)[0]
    sample = events[:, 0] - raw.first_samp
    codes = np.rint(flash[sample]).astype(int) - 2
    valid_structure = len(codes) % 96 == 0 and set(np.unique(codes)) == set(range(1, 13))
    for i, (event, code, label) in enumerate(zip(events, codes, labels)):
        block, offset = divmod(i, 96)
        segment = slice(block * 96, (block + 1) * 96)
        block_codes, block_labels = (codes[segment], labels[segment])
        structure = valid_structure and len(block_codes) == 96
        if structure:
            structure = all((set(block_codes[k:k + 12]) == set(range(1, 13)) for k in range(0, 96, 12)))
        positives = sorted(set(block_codes[block_labels == 1]))
        structure = structure and len(positives) == 2 and (positives[0] <= 6 < positives[1])
        if structure:
            structure = all((np.sum((block_codes == c) & (block_labels == 1)) == 8 for c in positives))
        first = events[block * 96, 0] / sfreq
        records.append({'character_id': f'{subject}:{session}:{run}:c{block:03d}', 'repetition': offset // 12 + 1, 'flash_code': int(code), 'character_true': (positives[0] - 1) * 6 + positives[1] - 7 if structure else -1, 'character_valid': bool(structure), 'n_choices': 36, 'flash_onset_seconds': event[0] / sfreq, 'character_onset_seconds': first, 'flash_duration_seconds': 0.125, 'last_interstimulus_interval_seconds': 0.125, 'selection_overhead_seconds': 2.0, 'timing_source': 'BNCI2014_009 published 125-ms stimulus and 2-s pretrial pause'})
    return pd.DataFrame(records)


def load_bnci(cfg):
    from moabb.datasets import BNCI2014_009
    root = Path(cfg.data_root) / 'BNCI'
    root.mkdir(parents=True, exist_ok=True)
    mne.set_config('MNE_DATA', str(root), set_env=True)
    dataset = BNCI2014_009()
    if len(dataset.subject_list) != 10:
        raise ValueError('Expected BNCI2014_009 with ten participants')
    raw_data = dataset.get_data(subjects=list(dataset.subject_list))
    target_entries = {k: v for k, v in dataset.event_id.items() if k.lower().replace('_', '').replace('-', '') == 'target'}
    if len(target_entries) != 1:
        raise ValueError(f'Cannot determine Target label from {dataset.event_id}')
    positive_id = next(iter(target_entries.values()))
    arrays, metadata = ([], [])
    for sid in dataset.subject_list:
        subject = f'S{int(sid):02d}'
        for session, runs in sorted(raw_data[sid].items()):
            for run_id, raw in sorted(runs.items()):
                stim = [c for c in raw.ch_names if c.lower() == 'target stim']
                if len(stim) != 1:
                    raise ValueError('BNCI009 raw Target stim channel is missing')
                events = mne.find_events(raw, stim_channel=stim[0], shortest_event=1, initial_event=True, verbose=False)
                events = events[np.isin(events[:, 2], list(dataset.event_id.values()))]
                y = (events[:, 2] == positive_id).astype(np.int64)
                character = extract_character_metadata(raw, events, y, subject, session, run_id)
                eeg = raw.copy().pick('eeg').load_data()
                if len(eeg.ch_names) != 16:
                    raise ValueError('BNCI009 requires its sixteen EEG electrodes')
                eeg.filter(0.1, 30.0, method='iir', iir_params={'order': 4, 'ftype': 'butter'}, verbose=False)
                epochs = mne.Epochs(eeg, events, event_id=dataset.event_id, tmin=cfg.p300_tmin, tmax=cfg.p300_tmax, baseline=None, preload=True, reject_by_annotation=True, verbose=False)
                kept = epochs.selection
                epochs.resample(cfg.fs, verbose=False)
                X = epochs.get_data(copy=True).astype(np.float32)
                rows = character.iloc[kept].copy().reset_index(drop=True)
                rows['y'] = y[kept]
                rows['subject'] = subject
                rows['recording_id'] = f'{subject}:{session}:{run_id}'
                rows['block_id'] = rows.get('character_id', pd.Series([f'{subject}:{session}:{run_id}'] * len(rows)))
                rows['onset_sample'] = events[kept, 0]
                rows['epoch_id'] = [f'BNCI:{subject}:{session}:{run_id}:{j}' for j in kept]
                arrays.append(X)
                metadata.append(rows)
                channels = tuple(eeg.ch_names)
    meta = pd.concat(metadata, ignore_index=True)
    return EEGData('BNCI', np.concatenate(arrays), meta.y.values.astype(np.int64), meta.subject.values.astype(str), meta, cfg.fs, channels, {'dataset': 'BNCI2014_009', 'population': 'healthy participants', 'loader': 'MOABB raw EEG + MNE epochs', 'filter_hz': [0.1, 30.0], 'raw_channel_unit': 'volts, including MOABB BNCI009 amplitude correction', 'dropped_epochs_are_retained_in_loader_selection': True}).validate()


def as_ucl_trials(value):
    if isinstance(value, np.ndarray) and value.dtype == object:
        return [t for x in value.flat for t in as_ucl_trials(x)]
    arr = np.asarray(value)
    if arr.ndim == 2:
        if arr.shape[-1] in (19, 22):
            return [arr[:, :19].T]
        if arr.shape[0] in (19, 22):
            return [arr[:19]]
        raise ValueError(f'UCL trial has no verified 19/22-channel axis: {arr.shape}')
    if arr.ndim == 3:
        axes = [i for i, n in enumerate(arr.shape) if n in (19, 22)]
        if len(axes) != 1:
            raise ValueError(f'Ambiguous UCL trial axes: {arr.shape}; provide explicit trial arrays')
        caxis = axes[0]
        remaining = [i for i in range(3) if i != caxis]
        taxis = max(remaining, key=lambda i: arr.shape[i])
        naxis = next((i for i in remaining if i != taxis))
        oriented = np.transpose(arr, (naxis, caxis, taxis))
        return [x[:19] for x in oriented]
    raise ValueError(f'Unsupported UCL array shape {arr.shape}')


def load_ucl(cfg):
    """Load eligible two-second UCL imagery trials with a sample-length audit.

    resample_poly returns ceil(raw_samples * up/down) samples. A trial that
    cannot supply the required window is excluded before filtering/resampling.
    Eligibility uses length only, identically for all subjects and classes.
    Eligible trials use the configured filter, units, resampler and two-second crop.
    """
    folder = Path(cfg.data_root) / 'UCL'
    audit_directory = Path(cfg.output_root) / 'audit'
    audit_directory.mkdir(parents=True, exist_ok=True)
    if cfg.download_public_data and not any(p.suffix.lower() == '.mat' for p in folder.rglob('*')):
        download_figshare(28156016, folder, version=1)
    files = sorted(p for p in folder.rglob('*') if p.suffix.lower() == '.mat')
    if not files:
        raise FileNotFoundError('No UCL MATLAB files found in ' + str(folder))
    manifest_path = Path(cfg.ucl_manifest_csv)
    manifest = pd.read_csv(manifest_path, dtype=str) if manifest_path.is_file() else None
    if manifest is not None:
        required = {'relative_path', 'subject', 'recording_id', 'sfreq', 'sfreq_source', 'units'}
        if not required.issubset(manifest.columns) or manifest[list(required)].isna().any().any():
            raise ValueError('UCL manifest has missing required fields.')
        if manifest.recording_id.duplicated().any():
            raise ValueError('UCL recording IDs must be unique.')
        source_rows = manifest.to_dict('records')
    else:
        source_rows = [{'relative_path': str(p.relative_to(folder))} for p in files]
    loaded, arrays, rows, inputs, used_manifest, eligibility = {}, [], [], [], [], []
    n_required = round(cfg.ucl_seconds * cfg.fs)
    observed_subjects = set()
    for entry in source_rows:
        path = (folder / entry['relative_path']).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file():
            raise ValueError('Invalid UCL input path: ' + str(path))
        if str(path) not in loaded:
            loaded[str(path)] = scipy_io.loadmat(path, squeeze_me=True, struct_as_record=False)
        contents = loaded[str(path)]
        records = list(ucl_structures(contents))
        if 'struct_path' in entry and pd.notna(entry['struct_path']):
            records = [(key, value) for key, value in records if '/'.join(key) == entry['struct_path']]
        if not records:
            raise ValueError('No verified L/R structures: ' + str(path))
        rate = ((float(entry['sfreq']), str(entry['sfreq_source']))
                if 'sfreq' in entry else ucl_sampling_metadata(contents))
        if rate is None and cfg.ucl_sfreq is not None:
            rate = float(cfg.ucl_sfreq), str(cfg.ucl_sfreq_source)
        if rate is None or not np.isfinite(rate[0]) or rate[0] <= 80 or not rate[1].strip():
            raise ValueError('UCL requires a verified sample rate and its source.')
        sfreq, rate_source = float(rate[0]), str(rate[1])
        ratio = Fraction(float(cfg.fs) / sfreq).limit_denominator(10000)
        unit = str(entry.get('units', cfg.ucl_units)).lower()
        scale = {'v': 1.0, 'uv': 1e-6, 'µv': 1e-6}.get(unit)
        if scale is None:
            raise ValueError('UCL units must be V or uV.')
        for key, structure in records:
            subject = str(entry['subject']) if 'subject' in entry else ucl_participant_id(key, path.stem)
            observed_subjects.add(subject)
            identity = '/'.join(key)
            recording = str(entry.get('recording_id', entry['relative_path'] + ':' + identity))
            if len(records) > 1:
                recording += ':' + identity
            used_manifest.append({'relative_path': entry['relative_path'], 'struct_path': identity,
                'subject': subject, 'recording_id': recording, 'sfreq': sfreq,
                'sfreq_source': rate_source, 'units': unit})
            counter = 0
            for kind, label in (('L', 0), ('R', 1)):
                for trial_index, raw_trial in enumerate(as_ucl_trials(structure[kind])):
                    epoch_id = f'UCL:{subject}:{recording}:{counter}'
                    block_id = recording + ':' + kind + ':' + str(counter // 10)
                    counter += 1  # Preserve stable trial identifiers across eligibility decisions.
                    raw_n = raw_trial.shape[-1]
                    expected_n = (raw_n * ratio.numerator + ratio.denominator - 1) // ratio.denominator
                    keep = expected_n >= n_required
                    eligibility.append({'subject': subject, 'y': label, 'trial_kind': kind,
                        'relative_path': entry['relative_path'], 'struct_path': identity,
                        'recording_id': recording, 'epoch_id': epoch_id, 'trial_index': trial_index,
                        'raw_samples': raw_n, 'raw_sfreq': sfreq,
                        'raw_duration_seconds': raw_n / sfreq,
                        'resampled_samples': expected_n, 'required_samples': n_required,
                        'kept': keep, 'reason': 'kept' if keep else 'shorter_than_analysis_window'})
                    if not keep:
                        continue
                    trial = fixed_bandpass((raw_trial * scale).astype(np.float32), sfreq, 4, 40)
                    trial = resample_array(trial, sfreq, cfg.fs)
                    if trial.shape[-1] != expected_n or trial.shape[-1] < n_required:
                        raise AssertionError('Resampler output length differs from the eligibility calculation.')
                    arrays.append(trial[:, :n_required])
                    rows.append({'y': label, 'subject': subject, 'recording_id': recording,
                        'block_id': block_id, 'trial_kind': kind, 'epoch_id': epoch_id, 'onset_sample': 0})
        inputs.append({'relative_path': entry['relative_path'], 'sha256': file_sha256(path),
                       'sfreq': sfreq, 'sfreq_source': rate_source, 'units': unit})
    audit = pd.DataFrame(eligibility)
    audit.to_csv(audit_directory / 'UCL_trial_eligibility.csv', index=False)
    excluded = audit[~audit.kept].copy()
    excluded.to_csv(audit_directory / 'UCL_excluded_short_trials.csv', index=False)
    summary = audit.groupby(['subject', 'trial_kind'], as_index=False).agg(
        n_total=('kept', 'size'), n_kept=('kept', 'sum'),
        minimum_raw_seconds=('raw_duration_seconds', 'min'),
        maximum_raw_seconds=('raw_duration_seconds', 'max'))
    summary['n_excluded'] = summary.n_total - summary.n_kept
    summary.to_csv(audit_directory / 'UCL_eligibility_counts.csv', index=False)
    pd.DataFrame(used_manifest).to_csv(audit_directory / 'UCL_manifest_used.csv', index=False)
    print('\nUCL length eligibility (same rule for every participant and class):', flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f'Kept {len(rows)}/{len(audit)} trials; excluded {len(excluded)} short trials.', flush=True)
    if len(observed_subjects) != 8:
        raise ValueError('Expected eight real UCL participants before exclusions.')
    if not arrays:
        raise ValueError('No UCL trials meet the configured two-second window. See the eligibility audit.')
    meta = pd.DataFrame(rows)
    if set(meta.subject.unique()) != observed_subjects:
        raise ValueError('A UCL participant has no eligible trials; no participant was silently dropped.')
    counts = meta.groupby(['subject', 'y']).size().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
    if (counts < 2).any().any():
        raise ValueError('Each of the eight participants needs at least two eligible trials per class. See the eligibility audit.')
    policy = {'version': UCL_ELIGIBILITY_POLICY, 'analysis_seconds': cfg.ucl_seconds,
        'analysis_sfreq': cfg.fs, 'required_samples': n_required,
        'rule': 'exclude only trials whose original resample_poly output cannot supply the configured window',
        'uses_labels_for_eligibility': False, 'padding': False,
        'n_total': len(audit), 'n_kept': len(rows), 'n_excluded_short': len(excluded)}
    atomic_json(audit_directory / 'UCL_exclusion_policy.json', policy)
    data = EEGData('UCL', np.stack(arrays), meta.y.to_numpy(dtype=np.int64),
        meta.subject.to_numpy(dtype=str), meta, cfg.fs, UCL_CHANNELS,
        {'inputs': inputs, 'task': 'left versus right imagery; rest excluded',
         'filter_hz': [4, 40], 'interval': 'first configured 2 s of eligible trials',
         'short_trial_policy': policy, 'eligibility_audit': str(audit_directory)}).validate()
    return data


def normalize_channel_name(name):
    import re
    name = re.sub('^EEG\\s*', '', str(name).strip(), flags=re.I).split('-')[0]
    return name.replace(' ', '').lower()


def eeget_identity(path):
    import re
    matches = []
    for part in Path(path).parts:
        if re.fullmatch('ALS[_-]?0?[1-6]', part, flags=re.I):
            matches.append(('ALS_{:02d}'.format(int(re.search('\\d+', part).group())), 1))
        elif re.fullmatch('id[_-]?\\d{1,3}', part, flags=re.I):
            matches.append(('id{:03d}'.format(int(re.search('\\d+', part).group())), 0))
    if len(set(matches)) != 1:
        raise ValueError(f'Cannot identify one ALS01–06 or id001–170 participant in {path}')
    return matches[0]


def load_eeget(cfg):
    folder = Path(cfg.data_root) / 'EEGET'
    if cfg.download_public_data and (not any((p.suffix.lower() == '.edf' for p in folder.rglob('*')))):
        download_figshare(24485689, folder)
    files = sorted([p for p in folder.rglob('*') if p.suffix.lower() == '.edf'])
    if not files:
        raise FileNotFoundError('Put the official EEGET EEG.edf recordings under data_root/EEGET. ET.csv is not EEG.')
    participant_files = {}
    for path in files:
        if path.stem.lower() != 'eeg':
            continue
        sid, label = eeget_identity(path)
        participant_files.setdefault((sid, label), []).append(path)
    arrays, rows, sources = ([], [], [])
    n_samples = round(cfg.fs * cfg.eeget_seconds)
    canonical_channels = None
    for (sid, label), paths in sorted(participant_files.items()):
        participant_trials, participant_rows = ([], [])
        for recording_index, path in enumerate(paths):
            raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
            normalized = [normalize_channel_name(c) for c in raw.ch_names]
            standard = mne.channels.make_standard_montage('standard_1020').ch_names
            montage_names = {normalize_channel_name(x) for x in standard}
            candidates = [i for i, c in enumerate(normalized) if c in montage_names]
            if len(candidates) != 32:
                raise ValueError(f'{path}: expected 32 named EEG electrodes, found {len(candidates)}. Provide a verified EDF channel-name mapping; do not select the first 32 columns.')
            order = sorted(candidates, key=lambda i: normalized[i])
            channels = tuple((normalized[i] for i in order))
            if canonical_channels is None:
                canonical_channels = channels
            elif channels != canonical_channels:
                raise ValueError('EEGET EDF channel sets disagree')
            raw.pick([raw.ch_names[i] for i in order]).load_data()
            original_fs = float(raw.info['sfreq'])
            raw.filter(0.5, min(40.0, original_fs / 2 - 1), method='iir', iir_params={'order': 4, 'ftype': 'butter'}, verbose=False)
            raw.resample(cfg.fs, verbose=False)
            X = raw.get_data().astype(np.float32)
            n_windows = X.shape[-1] // n_samples
            rec = str(path.relative_to(folder))
            for j in range(n_windows):
                participant_trials.append(X[:, j * n_samples:(j + 1) * n_samples])
                participant_rows.append({'y': label, 'subject': sid, 'recording_id': rec, 'block_id': f'{sid}:{rec}:block{j // 10}', 'onset_sample': j * n_samples, 'epoch_id': f'EEGET:{sid}:{rec}:{j}', 'recording_index': recording_index})
            sources.append({'relative_path': rec, 'subject': sid, 'n_windows_before_cap': n_windows, 'raw_sfreq': original_fs, 'sha256': file_sha256(path)})
        if not participant_trials:
            raise ValueError(f'No full EEG windows for {sid}')
        frame = pd.DataFrame(participant_rows)
        selected = []
        groups = [list(g.index) for _, g in frame.groupby('recording_id', sort=True)]
        for j in range(max(map(len, groups))):
            for g in groups:
                if j < len(g):
                    selected.append(g[j])
                if len(selected) >= cfg.eeget_window_cap:
                    break
            if len(selected) >= cfg.eeget_window_cap:
                break
        arrays.extend((participant_trials[i] for i in selected))
        rows.extend((participant_rows[i] for i in selected))
    meta = pd.DataFrame(rows)
    if meta.loc[meta.y == 1, 'subject'].nunique() != 6:
        raise ValueError('EEGET must contain all six ALS participants')
    return EEGData('EEGET', np.stack(arrays), meta.y.values.astype(np.int64), meta.subject.values.astype(str), meta, cfg.fs, canonical_channels, {'inputs': sources, 'modality': 'EDF EEG only', 'filter_hz': [0.5, 40.0], 'window_selection': 'nonoverlapping 2-s windows; chronological round-robin across recordings', 'window_cap': cfg.eeget_window_cap, 'label': 'ALS status, not intended character', 'acquisition_confound': 'ALS home/multiple sessions versus healthy lab recordings'}).validate()


@dataclass(frozen=True)
class Fold:
    dataset: str
    fold_id: str
    train_subjects: tuple
    val_subjects: tuple
    test_subjects: tuple
    origin: str
    protocol: str = 'source_only'

    def indices(self, data):
        groups = [set(self.train_subjects), set(self.val_subjects), set(self.test_subjects)]
        if any((a & b for a, b in itertools.combinations(groups, 2))):
            raise ValueError('Participant leakage in fold registry')
        arrays = tuple((np.flatnonzero(np.isin(data.subject, list(g))) for g in groups))
        for label, idx in zip(('train', 'val', 'test'), arrays):
            if not len(idx) or set(np.unique(data.y[idx])) != {0, 1}:
                raise ValueError(f'{data.name}/{self.fold_id}/{label}: missing binary class')
        return arrays


def balanced_source_subset(data, train, cfg):
    source_ids = sorted(np.unique(data.subject[train]))
    rng = np.random.default_rng(cfg.registry_seed + int(digest([data.name, source_ids])[:8], 16))
    n = min((np.sum(data.y[train] == c) for c in (0, 1)))
    selected = []
    for cls in (0, 1):
        subjects = sorted(np.unique(data.subject[train[data.y[train] == cls]]))
        rng.shuffle(subjects)
        pools = []
        for s in subjects:
            indices = train[(data.y[train] == cls) & (data.subject[train] == s)]
            blocks = sorted(data.meta.iloc[indices].block_id.unique())
            rng.shuffle(blocks)
            pieces = [rng.permutation(indices[data.meta.iloc[indices].block_id.values == b]) for b in blocks]
            ordered = [int(piece[j]) for j in range(max(map(len, pieces))) for piece in pieces if j < len(piece)]
            pools.append(np.array(ordered, dtype=int))
        group_selected = []
        for j in range(max(map(len, pools))):
            for pool in pools:
                if j < len(pool):
                    group_selected.append(int(pool[j]))
                if len(group_selected) == n:
                    break
            if len(group_selected) == n:
                break
        selected.extend(group_selected)
    return np.sort(np.array(selected, dtype=int))


def training_indices(data, fold, cfg):
    train, val, test = fold.indices(data)
    if data.name == 'EEGET' and cfg.balance_eeget_source:
        train = balanced_source_subset(data, train, cfg)
    return (train, val, test)


def ucl_structures(obj, path=()):
    if hasattr(obj, '_fieldnames'):
        obj = {k: getattr(obj, k) for k in obj._fieldnames}
    if isinstance(obj, dict):
        keys = {str(k).lower(): k for k in obj}
        if 'l' in keys and 'r' in keys:
            yield (path, {'L': obj[keys['l']], 'R': obj[keys['r']]})
        else:
            for k, value in obj.items():
                if not str(k).startswith('__'):
                    yield from ucl_structures(value, path + (str(k),))
    elif isinstance(obj, np.ndarray) and obj.dtype == object:
        for i, value in enumerate(obj.flat):
            yield from ucl_structures(value, path + (str(i),))
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            yield from ucl_structures(value, path + (str(i),))


def ucl_sampling_metadata(contents):
    found = []

    def walk(obj, path=()):
        if hasattr(obj, '_fieldnames'):
            obj = {k: getattr(obj, k) for k in obj._fieldnames}
        if isinstance(obj, dict):
            for key, value in obj.items():
                tag = ''.join((x for x in str(key).lower() if x.isalnum()))
                if tag in {'fs', 'sfreq', 'srate', 'samplingrate', 'samplingfrequency'}:
                    try:
                        a = np.asarray(value, dtype=float)
                        if a.size == 1 and 80 < float(a.item()) <= 10000:
                            found.append((float(a.item()), '/'.join(path + (str(key),))))
                    except (TypeError, ValueError):
                        pass
                elif not str(key).startswith('__') and tag not in {'l', 'r', 're'}:
                    walk(value, path + (str(key),))
        elif isinstance(obj, np.ndarray) and obj.dtype == object:
            for i, value in enumerate(obj.flat):
                walk(value, path + (str(i),))
    walk(contents)
    if len({x[0] for x in found}) > 1:
        raise ValueError('Conflicting UCL sampling-rate metadata: ' + str(found))
    return found[0] if found else None


def ucl_participant_id(struct_path, file_stem):
    import re
    known = set()
    for part in tuple(struct_path) + (str(file_stem),):
        match = re.fullmatch('(?:S|Subject|Participant)[_ -]*0*(\\d+)', str(part), flags=re.I)
        if match:
            known.add('S' + str(int(match.group(1))))
    if len(known) != 1:
        raise ValueError('Cannot verify one UCL participant from MAT keys and filename: ' + str((struct_path, file_stem)))
    return known.pop()


def dataset_display_name(name):
    return DATASET_DISPLAY_NAMES.get(name, name)


def display_subject(dataset, subject):
    subject = str(subject)
    if dataset == 'BNCI' and subject.startswith('S') and subject[1:].isdigit():
        return str(int(subject[1:]))
    return subject


def loader_code_signature(code):

    def constant(value):
        if isinstance(value, types.CodeType):
            return loader_code_signature(value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, tuple):
            return {'tuple': [constant(item) for item in value]}
        if isinstance(value, frozenset):
            return {'frozenset': sorted([constant(item) for item in value], key=canonical_json)}
        if isinstance(value, bytes):
            return {'bytes': value.hex()}
        if isinstance(value, complex):
            return {'complex': [value.real, value.imag]}
        return repr(value)
    return {'bytecode': code.co_code.hex(), 'constants': [constant(value) for value in code.co_consts], 'names': code.co_names, 'variables': code.co_varnames, 'closure': [code.co_cellvars, code.co_freevars], 'arguments': [code.co_argcount, code.co_kwonlyargcount], 'flags': code.co_flags}


def processed_data_contract(name, cfg):
    functions = {'BNCI': ('load_bnci', 'extract_character_metadata'), 'UCL': ('load_ucl', 'fixed_bandpass', 'resample_array', 'as_ucl_trials', 'ucl_structures', 'ucl_sampling_metadata', 'ucl_participant_id'), 'EEGET': ('load_eeget', 'normalize_channel_name', 'eeget_identity')}
    fields = {'BNCI': ('fs', 'p300_tmin', 'p300_tmax'), 'UCL': ('fs', 'ucl_seconds', 'ucl_sfreq', 'ucl_sfreq_source', 'ucl_units'), 'EEGET': ('fs', 'eeget_seconds', 'eeget_window_cap')}
    folder = Path(cfg.data_root) / name
    files = []
    for path in sorted(folder.rglob('*')) if folder.exists() else []:
        if path.is_file() and path.suffix.lower() not in {'.partial', '.tmp'}:
            stat = path.stat()
            files.append({'path': str(path.relative_to(folder)), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    manifest = None
    if name == 'UCL' and Path(cfg.ucl_manifest_csv).is_file():
        manifest = {'path': str(Path(cfg.ucl_manifest_csv).resolve()), 'sha256': file_sha256(cfg.ucl_manifest_csv)}
    versions = {}
    for package in ('numpy', 'scipy', 'pandas', 'mne', 'moabb'):
        try:
            versions[package] = package_metadata.version(package)
        except package_metadata.PackageNotFoundError:
            versions[package] = None
    definitions = {function: loader_code_signature(globals()[function].__code__) for function in functions[name]}
    definitions['EEGData.validate'] = loader_code_signature(EEGData.validate.__code__)
    return {'schema': 1, 'dataset': name, 'data_root': str(Path(cfg.data_root).resolve()), 'settings': {field: getattr(cfg, field) for field in fields[name]}, 'versions': versions, 'python': list(sys.version_info[:2]), 'loader_sha256': digest(definitions), 'channels': list(UCL_CHANNELS) if name == 'UCL' else None, 'inputs': files, 'manifest': manifest}


def processed_data_entry(name, cfg, contract):
    return Path(cfg.data_root) / 'processed_epochs' / name / digest(contract)


def read_processed_data(name, cfg, contract):
    directory = processed_data_entry(name, cfg, contract)
    record_path = directory / 'metadata.json'
    if not record_path.exists():
        return None
    record = json.loads(record_path.read_text())
    if record['contract'] != json.loads(canonical_json(contract)):
        raise ValueError('Processed data has a different preprocessing contract.')
    if set(record['payloads']) != {'X.npy', 'y.npy', 'subject.npy', 'trials.json'}:
        raise ValueError('Processed data has an incomplete payload manifest.')
    for filename, identity in record['payloads'].items():
        path = directory / filename
        stat = path.stat()
        if stat.st_size != identity['size'] or stat.st_mtime_ns != identity['mtime_ns']:
            raise ValueError('Processed data payload changed: ' + filename)
    X = np.load(directory / 'X.npy', mmap_mode='r', allow_pickle=False)
    y = np.load(directory / 'y.npy', allow_pickle=False)
    subjects = np.load(directory / 'subject.npy', allow_pickle=False)
    meta = pd.read_json(directory / 'trials.json', orient='table')
    d = EEGData(name, X, y, subjects, meta, record['fs'], tuple(record['channels']), record['provenance']).validate()
    if list(d.X.shape) != record['shape'] or str(d.X.dtype) != record['dtype'] or d.fs != cfg.fs or (not np.array_equal(d.y, d.meta.y.to_numpy(dtype=np.int64))):
        raise ValueError('Processed data labels or dimensions disagree with their manifest.')
    return d


def write_processed_data(d, cfg, contract):
    directory = processed_data_entry(d.name, cfg, contract)
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.epochs_', dir=directory.parent))
    try:
        for filename, values in (('X.npy', d.X), ('y.npy', d.y), ('subject.npy', d.subject.astype(str))):
            np.save(temporary / filename, values, allow_pickle=False)
        d.meta.to_json(temporary / 'trials.json', orient='table', index=False, double_precision=15)
        payloads = {}
        for filename in ('X.npy', 'y.npy', 'subject.npy', 'trials.json'):
            path = temporary / filename
            stat = path.stat()
            payloads[filename] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256_at_creation': file_sha256(path)}
        atomic_json(temporary / 'metadata.json', {'contract': contract, 'shape': list(d.X.shape), 'dtype': str(d.X.dtype), 'fs': d.fs, 'channels': list(d.channels), 'provenance': d.provenance, 'payloads': payloads, 'validation': 'validated original epochs; SHA256 recorded at creation; size/mtime, schema and EEG/label/participant checks on reuse'})
        if directory.exists():
            shutil.rmtree(directory)
        os.replace(temporary, directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return directory


def load_prepared_dataset(name, cfg, loader):
    start = time.perf_counter()
    if name == 'UCL':
        data = loader(cfg).validate()
        return data, {'status': 'preprocessed with trial-length audit', 'seconds': time.perf_counter() - start}
    contract = processed_data_contract(name, cfg)
    issue = None
    if cfg.processed_cache:
        try:
            cached = read_processed_data(name, cfg, contract)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issue = f'{type(exc).__name__}: {exc}'
            cached = None
        if cached is not None:
            return (cached, {'status': 'processed cache', 'seconds': time.perf_counter() - start, 'contract_sha256': digest(contract), 'cache_directory': str(processed_data_entry(name, cfg, contract)), 'rebuild_reason': None})
    d = loader(cfg).validate()
    directory = None
    if cfg.processed_cache:
        contract = processed_data_contract(name, cfg)
        directory = write_processed_data(d, cfg, contract)
    return (d, {'status': 'downloaded/preprocessed' if cfg.processed_cache else 'preprocessed; cache disabled', 'seconds': time.perf_counter() - start, 'contract_sha256': digest(contract), 'cache_directory': str(directory) if directory else None, 'rebuild_reason': issue})

class SourceChannelScaler:

    def fit(self, X):
        self.mean_ = X.mean(axis=(0, 2), keepdims=True, dtype=np.float64)
        self.scale_ = X.std(axis=(0, 2), keepdims=True, dtype=np.float64).clip(1e-08)
        return self

    def transform(self, X):
        return ((X - self.mean_) / self.scale_).astype(np.float32)


def covariance_features(X):
    centered = X - X.mean(axis=-1, keepdims=True)
    cov = np.einsum('nct,ndt->ncd', centered, centered) / max(1, X.shape[-1] - 1)
    c = X.shape[1]
    scale = np.trace(cov, axis1=1, axis2=2) / c
    cov += (0.001 * scale.clip(1e-12))[:, None, None] * np.eye(c)[None]
    cov /= np.trace(cov, axis1=1, axis2=2)[:, None, None].clip(1e-12)
    return cov[:, np.triu_indices(c)[0], np.triu_indices(c)[1]].astype(np.float32)


def base_views(X, fs, p300):
    spatial_windows = np.array_split(np.arange(X.shape[-1]), 4)
    windows = [np.arange(math.floor(k * X.shape[-1] / 4), math.ceil((k + 1) * X.shape[-1] / 4)) for k in range(4)] if p300 else spatial_windows
    temporal = np.concatenate([stat(X[:, :, w], axis=-1) for w in windows for stat in (np.mean, np.std, np.max, np.min)], axis=1)
    regions = np.array_split(np.arange(X.shape[1]), 4)
    spatial = np.concatenate([stat(X[:, r][:, :, w], axis=(1, 2))[:, None] for r in regions for w in spatial_windows for stat in (np.mean, np.std, np.max)], axis=1)
    bands = ((0.5, 4), (4, 8), (8, 13), (13, 30)) if p300 else ((4, 8), (8, 12), (12, 16), (16, 24), (24, 35))
    freq = np.fft.rfftfreq(X.shape[-1], 1 / fs)
    power = np.abs(np.fft.rfft(X, axis=-1)) ** 2 / X.shape[-1]
    spectral = []
    for low, high in bands:
        mask = (freq >= low) & (freq <= high)
        if not mask.any():
            raise ValueError('A configured spectral band has no FFT bins')
        spectral.append(np.log1p(power[:, :, mask].mean(axis=-1)))
    if p300:
        summary = []
        for w in windows:
            summary.extend([X[:, :, w].mean(axis=(1, 2))[:, None], X[:, :, w].std(axis=(1, 2))[:, None], np.abs(X[:, :, w]).max(axis=(1, 2))[:, None]])
        summary += [(X[:, :, np.concatenate(windows[1:])].mean(axis=(1, 2)) - X[:, :, windows[0]].mean(axis=(1, 2)))[:, None]]
        statistic = np.concatenate(summary, axis=1)
    else:
        statistic = np.concatenate((np.log(X.var(axis=-1) + 1e-06), X.mean(axis=-1), X.std(axis=-1), np.stack((X.mean(axis=(1, 2)), X.std(axis=(1, 2)), np.abs(X).max(axis=(1, 2))), axis=1)), axis=1)
    return {'temporal': temporal.astype(np.float32), 'spatial': spatial.astype(np.float32), 'spectral': np.concatenate(spectral, axis=1).astype(np.float32), 'statistic': statistic.astype(np.float32), 'covariance': covariance_features(X)}


class ViewPipeline:

    def __init__(self, cfg, dataset):
        self.cfg = cfg
        self.dataset = dataset

    def raw_features(self, X, fitting=False, y=None):
        views = base_views(X, self.cfg.fs, self.dataset == 'BNCI')
        if self.dataset == 'BNCI':
            if fitting:
                self.xdawn = XdawnCovariances(nfilter=4, estimator='oas', xdawn_estimator='oas')
                cov = self.xdawn.fit_transform(X, y)
                self.ts = TangentSpace(metric='riemann').fit(cov)
            else:
                cov = self.xdawn.transform(X)
            views['xdawn_ts'] = self.ts.transform(cov).astype(np.float32)
        else:
            cov = Covariances(estimator='oas').transform(X)
            if fitting:
                self.ts = TangentSpace(metric='riemann').fit(cov)
                self.csp = mne.decoding.CSP(n_components=min(6, X.shape[1] - 1), reg='ledoit_wolf', log=True, norm_trace=False, transform_into='average_power')
                self.csp.fit(X.astype(np.float64), y)
            views['oas_ts'] = self.ts.transform(cov).astype(np.float32)
            views['csp_logvar'] = self.csp.transform(X.astype(np.float64)).astype(np.float32)
        return views

    def fit(self, X, y, epoch_ids):
        self.fit_epoch_ids_hash = digest(list(epoch_ids))
        self.channel_scaler = SourceChannelScaler().fit(X)
        features = self.raw_features(self.channel_scaler.transform(X), fitting=True, y=y)
        self.reducers, self.dimension_rows = ({}, [])
        for name, values in features.items():
            cap = self.cfg.supervised_cap if name in {'xdawn_ts', 'oas_ts', 'csp_logvar'} else self.cfg.pca_cap
            apply_pca = values.shape[1] > cap and len(values) > cap
            n = min(cap, values.shape[1], len(values) - 1) if apply_pca else values.shape[1]
            steps = [('scale', StandardScaler())]
            if apply_pca:
                steps.append(('pca', PCA(n_components=n, svd_solver='full', whiten=False)))
            reducer = Pipeline(steps)
            reducer.fit(values)
            self.reducers[name] = reducer
            self.dimension_rows.append({'view': name, 'raw_dim': values.shape[1], 'input_dim': n, 'cap': cap, 'source_fit_n': len(X), 'fit_epoch_ids_hash': self.fit_epoch_ids_hash, 'supervised': name in {'xdawn_ts', 'csp_logvar'}})
        return self

    def transform(self, X):
        features = self.raw_features(self.channel_scaler.transform(X))
        return {k: self.reducers[k].transform(v).astype(np.float32) for k, v in features.items()}

    def raw_transform(self, X):
        return self.channel_scaler.transform(X)

class QuantumCore(nn.Module):

    def __init__(self, q, depth, kind='mvqan'):
        super().__init__()
        self.q, self.depth, self.kind = (q, depth, kind)
        shape = (depth, q, 2) if kind == 'mvqan' else (depth, q)
        self.weights = nn.Parameter(torch.empty(shape).uniform_(-0.1, 0.1))
        device = qml.device('default.qubit', wires=q, shots=None)

        @qml.qnode(device, interface='torch', diff_method='backprop')
        def circuit(inputs, weights):
            state = torch.zeros(2 ** q, dtype=torch.complex128, device=inputs.device)
            state[0] = 1.0
            qml.StatePrep(state, wires=range(q))
            for j in range(q):
                qml.RY(inputs[..., j], wires=j)
                if kind == 'mvqan':
                    qml.RZ(inputs[..., q + j], wires=j)
            for layer in range(depth):
                if kind == 'qeegnet':
                    for j in range(q):
                        qml.CNOT(wires=[j, (j + 1) % q])
                for j in range(q):
                    qml.RY(weights[..., layer, j, 0] if kind == 'mvqan' else weights[..., layer, j], wires=j)
                    if kind == 'mvqan':
                        qml.RZ(weights[..., layer, j, 1], wires=j)
                if kind == 'mvqan':
                    for j in range(q):
                        qml.CNOT(wires=[j, (j + 1) % q])
            return [qml.expval(qml.PauliZ(j)) for j in range(q)]
        self.qnode = circuit

    def forward(self, angles):
        values = self.qnode(angles, self.weights)
        result = torch.stack(values, dim=-1)
        return result.to(device=angles.device, dtype=angles.dtype)


class PrincipalEncoder(nn.Module):

    def __init__(self, dim, q, depth, dropout, kind='VQC', width=None, core_k=3, seed=1701):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.kind = kind
        h = width or max(32, 4 * q)
        self.front = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, 2 * q))
        if kind != 'VQC':
            raise ValueError('This package exposes only the requested VQC branch.')
        self.core = QuantumCore(q, depth)
        self.post = nn.Sequential(nn.LayerNorm(q), nn.Linear(q, q), nn.GELU(), nn.LayerNorm(q))

    def forward(self, x):
        encoded = self.front(self.norm(x))
        encoded = self.core(math.pi * torch.tanh(encoded))
        return self.post(encoded)


class ResidualEncoder(nn.Module):

    def __init__(self, dim, out, dropout, width=None):
        super().__init__()
        h = width or max(32, 4 * out)
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, out), nn.LayerNorm(out))

    def forward(self, x):
        return self.net(x)


class MultiViewNetwork(nn.Module):

    def __init__(self, dims, cfg, spec):
        super().__init__()
        self.view_names = tuple(dims)
        self.q = spec.get('q', cfg.q)
        self.depth = spec.get('depth', cfg.depth)
        self.spec = spec
        self.fused_quantum = cfg.fuse_quantum_views and spec.get('encoder', 'VQC') == 'VQC' and (len(dims) > 1)
        self.temperature = cfg.attention_temperature
        self.fusion = spec.get('fusion', 'attention')
        kind = spec.get('encoder', 'VQC')
        self.use_residual = spec.get('residual', True)
        dropout = spec.get('dropout', cfg.dropout)
        residual_dim = spec.get('residual_dim', cfg.residual_dim)
        self.encoders = nn.ModuleDict({k: PrincipalEncoder(d, self.q, self.depth, dropout, kind, spec.get('widths', {}).get(k), spec.get('core_k', 3), seed=1701 + i) for i, (k, d) in enumerate(dims.items())})
        self.residuals = nn.ModuleDict({k: ResidualEncoder(d, residual_dim, dropout, spec.get('residual_width')) for k, d in dims.items()}) if self.use_residual else nn.ModuleDict()
        h = max(8, 2 * self.q)
        self.scorer = nn.Sequential(nn.Linear(self.q, h), nn.Tanh(), nn.Linear(h, 1)) if len(dims) > 1 and self.fusion == 'attention' else None
        width = max(16, 4 * self.q)
        self.classifier = nn.Sequential(nn.Linear(self.q + (residual_dim if self.use_residual else 0), width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 2))

    def forward(self, views):
        if self.fused_quantum:
            encoders = [self.encoders[k] for k in self.view_names]
            angles = torch.stack([math.pi * torch.tanh(e.front(e.norm(views[k]))) for k, e in zip(self.view_names, encoders)], dim=1)
            b, m, a = angles.shape
            independent_weights = torch.stack([e.core.weights for e in encoders])
            weights = independent_weights.unsqueeze(0).expand(b, -1, -1, -1, -1).reshape(b * m, self.depth, self.q, 2)
            raw_z = torch.stack(encoders[0].core.qnode(angles.reshape(b * m, a), weights), dim=-1)
            raw_z = raw_z.to(dtype=angles.dtype, device=angles.device).reshape(b, m, self.q)
            Z = torch.stack([e.post(raw_z[:, i]) for i, e in enumerate(encoders)], dim=1)
        else:
            Z = torch.stack([self.encoders[k](views[k]) for k in self.view_names], dim=1)
        alpha = torch.full(Z.shape[:2], 1 / len(self.view_names), device=Z.device, dtype=Z.dtype) if self.scorer is None else torch.softmax(self.scorer(Z).squeeze(-1) / self.temperature, dim=1)
        hq = (alpha.unsqueeze(-1) * Z).sum(dim=1)
        h = hq
        if self.use_residual:
            R = torch.stack([self.residuals[k](views[k]) for k in self.view_names], dim=1)
            h = torch.cat((hq, (alpha.unsqueeze(-1) * R).sum(dim=1)), dim=1)
        return {'logits': self.classifier(h), 'hq': hq, 'attention': alpha, 'per_view': Z, 'fused': h}

def metrics_from_probs(y, prob, threshold=None):
    """prob: (n, K) class probabilities. Binary uses the source-selected threshold."""
    y = np.asarray(y, int)
    K = prob.shape[1]
    if K == 2:
        thr = 0.5 if threshold is None else threshold
        pred = (prob[:, 1] >= thr).astype(int)
        auroc = roc_auc_score(y, prob[:, 1])
        auprc = average_precision_score(y, prob[:, 1])
    else:
        pred = prob.argmax(1)
        auroc = roc_auc_score(y, prob, multi_class='ovr', average='macro', labels=list(range(K)))
        auprc = float(np.mean([average_precision_score(y == k, prob[:, k]) for k in range(K)]))
    out = {
        'macro_f1': f1_score(y, pred, average='macro', labels=list(range(K)), zero_division=0),
        'bacc': balanced_accuracy_score(y, pred),
        'acc': accuracy_score(y, pred),
        'auroc': auroc, 'auprc': auprc,
        'threshold': None if K > 2 else float(0.5 if threshold is None else threshold),
        'single_class_predictions': bool(len(np.unique(pred)) == 1),
        'score_std': float(prob.std()),
    }
    cm = confusion_matrix(y, pred, labels=list(range(K)))
    out['confusion'] = cm.tolist()
    # identity checks (per run): balanced test set -> Acc == BAcc and MF1 <= BAcc (binary)
    counts = np.bincount(y, minlength=K)
    if K == 2 and counts[0] == counts[1]:
        assert abs(out['acc'] - out['bacc']) < 1e-9, 'balanced-set identity violated'
        assert out['macro_f1'] <= out['bacc'] + 1e-9, 'MF1 > BAcc on balanced set'
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def select_threshold_val(y_val, prob_val):
    if prob_val.shape[1] != 2:
        return None
    grid = np.round(np.linspace(0.01, 0.99, 99), 2)
    keyed = [((f1_score(y_val, (prob_val[:, 1] >= t).astype(int), average='macro', zero_division=0),
               -abs(t - 0.5), -t), t) for t in grid]
    return float(max(keyed)[1])


def val_criterion(y_val, prob_val):
    thr = select_threshold_val(y_val, prob_val)
    m = metrics_from_probs(y_val, prob_val, thr)
    return (m['macro_f1'], m['auroc']), thr


def class_weights(y, K, positive_multiplier=1.25):
    counts = np.bincount(y, minlength=K).astype(float)
    w = len(y) / (K * np.maximum(counts, 1))
    if K == 2:
        w[1] *= positive_multiplier
    return torch.tensor(w, dtype=torch.float32)


def focal_loss(logits, y, w, gamma=2.0):
    ce = F.cross_entropy(logits, y, weight=w, reduction='none')
    return ((1 - torch.exp(-ce)).pow(gamma) * ce).mean()


def _cov(z):
    c = z - z.mean(0, keepdim=True)
    return c.T @ c / max(1, len(z) - 1)


def conditional_moment_loss(z, y, d, stats=None):
    """Eq. 6: mean over valid (subject,class) groups of ||mu_sk-mu_k||^2/Q + ||S_sk-S_k||_F^2/Q^2."""
    terms, Q = [], z.shape[1]
    for k in torch.unique(y):
        ref = z[y == k]
        if len(ref) < 2:
            continue
        mu_k, S_k = ref.mean(0), _cov(ref)
        for s in torch.unique(d[y == k]):
            g = z[(y == k) & (d == s)]
            if len(g) < 2:
                continue
            terms.append(((g.mean(0) - mu_k) ** 2).sum() / Q + ((_cov(g) - S_k) ** 2).sum() / Q ** 2)
            if stats is not None:
                stats['n_sk'].append(len(g))
    if stats is not None:
        stats['valid_groups'].append(len(terms))
        stats['subjects_per_batch'].append(int(torch.unique(d).numel()))
    return torch.stack(terms).mean() if terms else z.sum() * 0


def separation_loss_eq7(z, y, margin=0.75):
    z = F.normalize(z, dim=1)
    centers, intra = [], []
    for k in torch.unique(y):
        g = z[y == k]
        if len(g) >= 2:
            c = g.mean(0)
            centers.append(c)
            intra.append(((g - c) ** 2).sum(1).mean())
    if not intra:
        return z.sum() * 0
    inter = [F.relu(margin - torch.linalg.vector_norm(a - b)) ** 2
             for i, a in enumerate(centers) for b in centers[i + 1:]]
    return torch.stack(intra).mean() + (torch.stack(inter).mean() if inter else z.sum() * 0)


class SubjectClassSampler(torch.utils.data.Sampler):
    def __init__(self, y, d, batch_size, seed, per_group=4):
        self.y, self.d = np.asarray(y), np.asarray(d)
        self.K = int(self.y.max()) + 1
        self.bs, self.seed, self.per_group, self.epoch = batch_size, seed, per_group, 0
        self.groups = {(int(s), k): np.flatnonzero((self.d == s) & (self.y == k))
                       for s in np.unique(self.d) for k in range(self.K)
                       if np.any((self.d == s) & (self.y == k))}

    def __len__(self):
        return max(1, math.ceil(len(self.y) / self.bs))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 104729 * self.epoch)
        pools = {k: [g for g in self.groups if g[1] == k] for k in range(self.K)}
        for k in pools:
            rng.shuffle(pools[k])
        offs = {k: 0 for k in range(self.K)}
        quota = self.bs // self.K
        for _ in range(len(self)):
            batch = []
            for k in range(self.K):
                chosen = set()
                n_avail = int(np.sum(self.y == k))
                while len(chosen) < min(quota, n_avail):
                    key = pools[k][offs[k] % len(pools[k])]
                    offs[k] += 1
                    pool = np.array([i for i in self.groups[key] if i not in chosen])
                    n = min(self.per_group, len(pool), quota - len(chosen))
                    if n > 0:
                        chosen.update(rng.choice(pool, n, replace=False).tolist())
                batch.extend(sorted(chosen))
            rng.shuffle(batch)
            yield batch


class _DS(torch.utils.data.Dataset):
    def __init__(self, X, y, d):
        self.X, self.y, self.d = X, torch.as_tensor(y).long(), torch.as_tensor(d).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = {k: v[i] for k, v in self.X.items()} if isinstance(self.X, dict) else self.X[i]
        return x, self.y[i], self.d[i]


def _to(x, dev):
    return {k: v.to(dev) for k, v in x.items()} if isinstance(x, dict) else x.to(dev)


def _subset(X, idx):
    if isinstance(X, dict):
        return {k: torch.as_tensor(np.ascontiguousarray(v[idx])) for k, v in X.items()}
    return torch.as_tensor(np.ascontiguousarray(X[idx]))


def _out(o):
    if isinstance(o, dict):
        return o
    return {'logits': o, 'hq': o}


def n_trainable(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _set_n_classes(model, K):
    last = model.classifier[-1]
    if last.out_features != K:
        model.classifier[-1] = nn.Linear(last.in_features, K)
    return model


def build_mvqan(dims, cfg, K):
    return _set_n_classes(MultiViewNetwork(dims, cfg, {'encoder': 'VQC'}), K)


class ResidualOnlyNetwork(nn.Module):
    """MVQAN with the quantum (principal) branch removed.
    View weights are computed from the residual embeddings (no VQC to condition on);
    alignment and separation act on the fused residual representation h^r."""

    def __init__(self, dims, cfg, K, residual_dim=16, residual_width=None, dropout=0.2):
        super().__init__()
        self.view_names = tuple(dims)
        self.residuals = nn.ModuleDict({k: ResidualEncoder(d, residual_dim, dropout, residual_width) for k, d in dims.items()})
        h = max(8, 2 * cfg.q)
        self.scorer = nn.Sequential(nn.Linear(residual_dim, h), nn.Tanh(), nn.Linear(h, 1))
        self.temperature = cfg.attention_temperature
        width = max(16, 4 * cfg.q)
        self.classifier = nn.Sequential(nn.Linear(residual_dim, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, K))

    def forward(self, views):
        R = torch.stack([self.residuals[k](views[k]) for k in self.view_names], dim=1)
        alpha = torch.softmax(self.scorer(R).squeeze(-1) / self.temperature, dim=1)
        h = (alpha.unsqueeze(-1) * R).sum(1)
        return {'logits': self.classifier(h), 'hq': h, 'attention': alpha}


def build_residual_only(dims, cfg, K, matched=False):
    if not matched:
        return ResidualOnlyNetwork(dims, cfg, K, residual_dim=cfg.residual_dim)
    target = n_trainable(build_mvqan(dims, cfg, K))
    best = None
    for width in range(16, 2049):
        m = ResidualOnlyNetwork(dims, cfg, K, residual_dim=cfg.residual_dim, residual_width=width)
        gap = abs(n_trainable(m) - target) / target
        if best is None or gap < best[0]:
            best = (gap, width)
        if n_trainable(m) > target * 1.02:
            break
    if best[0] >= REB.matched_tolerance:
        raise ValueError(f'Residual-only: no width within 1% (best {best[0]:.3%})')
    return ResidualOnlyNetwork(dims, cfg, K, residual_dim=cfg.residual_dim, residual_width=best[1])


def _sym(x):
    return 0.5 * (x + x.transpose(-1, -2))


def triu_vec(S):
    n = S.shape[-1]
    i, j = torch.triu_indices(n, n, device=S.device)
    w = torch.where(i == j, torch.ones_like(i, dtype=S.dtype), torch.full_like(i, math.sqrt(2), dtype=S.dtype))
    return S[..., i, j] * w


class BiMap(nn.Module):
    """W^T S W with W on the Stiefel manifold (orthonormal columns via QR re-parameterisation)."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.A = nn.Parameter(torch.linalg.qr(torch.randn(n_in, n_out))[0])

    def forward(self, S):
        W = torch.linalg.qr(self.A.to(S.dtype))[0]
        return W.transpose(-1, -2) @ S @ W


class ReEig(nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, S):
        return spd_fn(S, lambda v: v.clamp_min(self.eps))


class SPDNet(nn.Module):
    """Huang & Van Gool (2017): Cov -> BiMap -> ReEig -> BiMap -> ReEig -> LogEig -> linear."""

    def __init__(self, C, K, d1=16, d2=8):
        super().__init__()
        d1 = min(d1, C); d2 = min(d2, d1)
        self.net = nn.Sequential(BiMap(C, d1), ReEig(), BiMap(d1, d2), ReEig())
        self.fc = nn.Linear(d2 * (d2 + 1) // 2, K)

    def forward(self, X):
        z = triu_vec(logm(self.net(batch_cov(X.double())))).float()
        return {'logits': self.fc(z), 'hq': z}


def euclidean_alignment(X, subjects):
    Xa = np.empty_like(X)
    for s in np.unique(subjects):
        idx = np.flatnonzero(subjects == s)
        Xi = X[idx].astype(np.float64)
        R = np.mean([x @ x.T / x.shape[-1] for x in Xi], axis=0)
        vals, vecs = np.linalg.eigh(R)
        Rm = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-12))) @ vecs.T
        Xa[idx] = np.einsum('cd,ndt->nct', Rm, Xi).astype(X.dtype)
    return Xa


def canonical_channel_names(names):
    import mne as _mne
    ref = {n.lower(): n for n in _mne.channels.make_standard_montage('standard_1005').ch_names}
    out = []
    for n in names:
        key = str(n).strip().lower()
        if key not in ref:
            raise ValueError(f'Channel {n} not in standard_1005; supply a mapping before using foundation models')
        out.append(ref[key])
    return out


def fm_inputs(ctx, fs, name):
    """Volts -> units of 100 uV (LaBraM/CBraMod convention); resample to 200 Hz; zero-pad to whole 1-s patches.
    REVE: per-trial channel z-score clipped at 15 SD (REVE pretraining convention)."""
    X = ctx['unscaled'].astype(np.float64)
    X = sp_signal.resample_poly(X, FM_SFREQ, int(fs), axis=-1)
    T = X.shape[-1]
    T_pad = int(math.ceil(T / FM_SFREQ) * FM_SFREQ)
    if T_pad > T:
        X = np.concatenate([X, np.zeros(X.shape[:-1] + (T_pad - T,))], axis=-1)
    if name == 'REVE':
        mu, sd = X.mean(-1, keepdims=True), X.std(-1, keepdims=True) + 1e-12
        X = np.clip((X - mu) / sd, -15, 15)
    else:
        X = X * 1e4
    return X.astype(np.float32)

# Experiment models and training routines.

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=lambda x:
        x.tolist() if isinstance(x, np.ndarray) else x.item() if isinstance(x, np.generic)
        else str(x)))
    os.replace(temporary, path)


def atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


class _SymmetricFunction(torch.autograd.Function):
    """Spectral matrix function with divided-difference derivatives.

    Unlike differentiating eigenvectors directly, this derivative has a finite
    limit at repeated eigenvalues. Used for ReEig, LogEig, exp and inverse sqrt.
    """

    @staticmethod
    def forward(ctx, S, fn, eps):
        values, vectors = torch.linalg.eigh(_sym(S))
        with torch.enable_grad():
            eigenvalues = values.detach().requires_grad_(True)
            mapped = fn(eigenvalues if eps is None else eigenvalues.clamp_min(eps))
            derivative = torch.autograd.grad(mapped.sum(), eigenvalues)[0]
        mapped = mapped.detach()
        ctx.save_for_backward(values, vectors, mapped, derivative.detach())
        return (vectors * mapped.unsqueeze(-2)) @ vectors.transpose(-1, -2)

    @staticmethod
    def backward(ctx, grad):
        values, vectors, mapped, derivative = ctx.saved_tensors
        delta = values.unsqueeze(-1) - values.unsqueeze(-2)
        scale = values.abs().amax(-1, keepdim=True).unsqueeze(-1).clamp_min(1)
        close = delta.abs() < 1e-10 * scale
        den = torch.where(close, torch.ones_like(delta), delta)
        divided = (mapped.unsqueeze(-1) - mapped.unsqueeze(-2)) / den
        limit = (derivative.unsqueeze(-1) + derivative.unsqueeze(-2)) * 0.5
        loewner = torch.where(close, limit, divided)
        local = vectors.transpose(-1, -2) @ _sym(grad) @ vectors
        result = vectors @ (loewner * local) @ vectors.transpose(-1, -2)
        return _sym(result), None, None


def spd_fn(S, fn, eps=1e-6):
    return _SymmetricFunction.apply(S, fn, eps)


def logm(S):
    return spd_fn(S, torch.log)


def expm(S):
    return spd_fn(S, torch.exp, eps=None)


def invsqrtm(S):
    return spd_fn(S, torch.rsqrt)


def batch_cov(X, shrink=1e-3):
    if X.shape[-1] < 2:
        raise ValueError('Covariance needs at least two time samples.')
    centered = X - X.mean(-1, keepdim=True)
    S = centered @ centered.transpose(-1, -2) / (X.shape[-1] - 1)
    scale = S.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)
    eye = torch.eye(S.shape[-1], dtype=S.dtype, device=S.device)
    return _sym(S) + shrink * scale[..., None, None] * eye


class MAttStyle(nn.Module):
    """MAtt reimplementation adapted to the input channel/class counts.

    Spatial/temporal convolution -> segment covariances -> Q/K/V BiMaps ->
    squared log-Euclidean distance attention -> log-Euclidean mean -> ReEig ->
    LogEig -> classifier. Algorithm reference: https://github.com/CECNL/MAtt .
    Unequal-length segments become fixed-size SPD matrices BEFORE stacking.
    """

    def __init__(self, C, T, K, n_segments=3, conv_ch=20, spd_out=18):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(1, C, (C, 1)), nn.BatchNorm2d(C))
        self.c2 = nn.Sequential(nn.Conv2d(C, conv_ch, (1, 12), padding=(0, 6)),
                                nn.BatchNorm2d(conv_ch))
        self.n_seg = n_segments
        self.q = BiMap(conv_ch, spd_out)
        self.k = BiMap(conv_ch, spd_out)
        self.v = BiMap(conv_ch, spd_out)
        self.fc = nn.Linear(n_segments * spd_out * (spd_out + 1) // 2, K)

    def forward(self, X):
        h = self.c2(self.c1(X.unsqueeze(1))).squeeze(2).double()
        covariances = []
        for segment in torch.tensor_split(h, self.n_seg, dim=-1):
            centered = segment - segment.mean(-1, keepdim=True)
            S = centered @ centered.transpose(-1, -2) / (segment.shape[-1] - 1)
            tr = S.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-12)
            S = S / tr[..., None, None]
            S = S + 1e-5 * torch.eye(S.shape[-1], device=S.device, dtype=S.dtype)
            covariances.append(S)
        S = torch.stack(covariances, dim=1)
        LQ, LK, LV = logm(self.q(S)), logm(self.k(S)), logm(self.v(S))
        squared_distance = (LQ.unsqueeze(2) - LK.unsqueeze(1)).square().sum((-2, -1))
        attention = torch.softmax(1 / (1 + torch.log1p(squared_distance)), dim=-1)
        mean_log = torch.einsum('bij,bjmn->bimn', attention, LV)
        result = ReEig()(expm(mean_log))
        z = triu_vec(logm(result)).flatten(1).float()
        return {'logits': self.fc(z), 'hq': z}


def spd_geodesic(A, B, fraction):
    root = spd_fn(A, torch.sqrt)
    inverse = invsqrtm(A)
    return _sym(root @ expm(fraction * logm(inverse @ B @ inverse)) @ root)


def karcher_mean(S, steps=1):
    mean = S.mean(0, keepdim=True)
    for _ in range(steps):
        root = spd_fn(mean.detach(), torch.sqrt)
        inverse = invsqrtm(mean.detach())
        tangent = logm(inverse @ S @ inverse).mean(0, keepdim=True)
        mean = _sym(root @ expm(tangent) @ root)
    return mean


def spd_variance(S, mean):
    inverse = invsqrtm(mean)
    return logm(inverse @ S @ inverse).square().sum((-2, -1)).mean()


class SPDDSMBN(nn.Module):
    """Domain-specific momentum BN on SPD matrices (affine-invariant metric).

    Per-domain Riemannian means and scalar dispersions; shared learned std;
    fixed identity output mean, as in the TSMNet SPDDSMBN configuration.
    Normalizing at the identity followed by LogEig is computed directly in
    its tangent space. Target statistics are fitted separately for each
    unlabeled participant, including the two EEGET test participants.
    Reference equations/algorithm: https://github.com/rkobler/TSMNet .
    """

    def __init__(self, dimension, n_domains):
        super().__init__()
        identities = torch.eye(dimension, dtype=torch.float64).repeat(n_domains, 1, 1)
        self.register_buffer('running_mean', identities.clone())
        self.register_buffer('test_mean', identities.clone())
        self.register_buffer('running_var', torch.ones(n_domains, dtype=torch.float64))
        self.register_buffer('test_var', torch.ones(n_domains, dtype=torch.float64))
        self.register_buffer('fitted', torch.zeros(n_domains, dtype=torch.bool))
        self.std = nn.Parameter(torch.ones((), dtype=torch.float64))
        self.eta, self.eta_test = 1.0, 0.1

    def set_epoch(self, epoch, epochs, batch_size):
        # Adaptive-momentum schedule; source statistics never use test labels.
        ratio = min(batch_size, 64) / 64
        rho = ratio ** (1 / max(1, epochs))
        weight = max(epochs - epoch - 1, 0) / max(1, epochs - 1)
        self.eta = 1 - (rho ** (epochs * weight) - rho ** epochs)
        self.eta_test = 1 - 0.9 ** (ratio * (1 - weight) + weight)

    @torch.no_grad()
    def fit_unlabeled(self, S, domains):
        for d in torch.unique(domains):
            index = int(d)
            group = S[domains == d]
            mean = karcher_mean(group, steps=15)
            self.test_mean[index].copy_(mean.squeeze(0))
            self.test_var[index].copy_(spd_variance(group, mean).clamp_min(1e-7))
            self.fitted[index] = True

    def forward(self, S, domains):
        result = torch.empty_like(S)
        for d in torch.unique(domains):
            index = int(d)
            selected = domains == d
            group = S[selected]
            if self.training:
                batch_mean = karcher_mean(group, steps=1)
                mean = spd_geodesic(self.running_mean[index].detach().clone().unsqueeze(0),
                                    batch_mean, self.eta)
                var = ((1 - self.eta) * self.running_var[index].detach().clone()
                       + self.eta * spd_variance(group, mean))
                with torch.no_grad():
                    self.running_mean[index].copy_(mean.squeeze(0))
                    self.running_var[index].copy_(var)
                    new_test = spd_geodesic(self.test_mean[index].clone().unsqueeze(0),
                                             batch_mean.detach(), self.eta_test)
                    self.test_mean[index].copy_(new_test.squeeze(0))
                    self.test_var[index].mul_(1 - self.eta_test).add_(
                        self.eta_test * spd_variance(group.detach(), new_test))
                    self.fitted[index] = True
            else:
                if not self.fitted[index]:
                    raise RuntimeError('Fit the unlabeled evaluation domain statistics first.')
                mean = self.test_mean[index].unsqueeze(0)
                var = self.test_var[index]
            inverse = invsqrtm(mean)
            # Log[(mean^-1/2 S mean^-1/2)^(std/sqrt(var))].
            result[selected] = logm(inverse @ group @ inverse) * self.std / (var + 1e-5).sqrt()
        return result


class TSMNetSPDDSMBN(nn.Module):
    """TSMNet architecture with affine-invariant SPDDSMBN, implemented here.

    Uses QR-reparameterized BiMap and the common source-only trainer in this
    package. This is a reimplementation, not an execution of the authors' code.
    """

    def __init__(self, C, T, K, n_domains):
        super().__init__()
        self.temporal = nn.Conv2d(1, 4, (1, 25), padding=(0, 12), padding_mode='reflect')
        self.spatial = nn.Conv2d(4, 40, (C, 1))
        self.bimap, self.reeig = BiMap(40, 20), ReEig()
        self.bn = SPDDSMBN(20, n_domains)
        self.fc = nn.Linear(210, K)
        self.current_domains = None
        self.target_bias = None

    def covariance(self, X):
        h = self.spatial(self.temporal(X.unsqueeze(1))).squeeze(2).double()
        return self.reeig(self.bimap(batch_cov(h)))

    def forward(self, X, domains=None):
        domains = self.current_domains if domains is None else domains
        if domains is None:
            raise ValueError('TSMNet requires participant/domain indices for each trial.')
        z = triu_vec(self.bn(self.covariance(X), domains)).float()
        if self.target_bias is not None:
            z = z + self.target_bias
        return {'logits': self.fc(z), 'hq': z}

    @torch.no_grad()
    def prepare_eval_domain(self, X, device, domains):
        was_training = self.training
        self.eval()
        covariances = torch.cat([self.covariance(_to(X[i:i + 64], device))
                                 for i in range(0, len(X), 64)])
        self.bn.fit_unlabeled(covariances, torch.as_tensor(domains, device=device))
        self.train(was_training)


@torch.no_grad()
def predict_probs(model, X, device, bs=32, forward_kwargs=None, domains=None):
    model.eval()
    n = len(next(iter(X.values()))) if isinstance(X, dict) else len(X)
    outputs = []
    for start in range(0, n, bs):
        idx = slice(start, start + bs)
        xb = {k: v[idx] for k, v in X.items()} if isinstance(X, dict) else X[idx]
        kwargs = dict(forward_kwargs or {})
        if isinstance(model, TSMNetSPDDSMBN):
            kwargs['domains'] = torch.as_tensor(domains[idx], device=device)
        logits = _out(model(_to(xb, device), **kwargs))['logits']
        if logits.ndim != 2 or logits.shape[-1] != 2 or not torch.isfinite(logits).all():
            raise ValueError('Expected finite (batch, 2) classification logits.')
        outputs.append(logits.softmax(1).float().cpu().numpy())
    return np.concatenate(outputs)


def spdim_adapt(model, X, domains, device, steps, lr, directory):
    """Fixed-step information maximization; update tangent bias only.

    No target labels, target validation, early stopping or target threshold tuning.
    This controlled variant is explicitly reported as SPDIM-style.
    """
    model.prepare_eval_domain(X, device, domains)
    model.eval()
    vectors = []
    with torch.no_grad():
        for start in range(0, len(X), 64):
            d = torch.as_tensor(domains[start:start + 64], device=device)
            vectors.append(model(_to(X[start:start + 64], device), domains=d)['hq'])
    z = torch.cat(vectors).detach()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.target_bias = nn.Parameter(torch.zeros(z.shape[-1], device=device))
    optimizer = torch.optim.Adam([model.target_bias], lr=lr)
    history = []
    for step in range(steps):
        p = model.fc(z + model.target_bias).softmax(1)
        conditional_entropy = -(p * torch.log(p.clamp_min(1e-8))).sum(1).mean()
        mean_prob = p.mean(0)
        marginal_entropy = -(mean_prob * torch.log(mean_prob.clamp_min(1e-8))).sum()
        loss = conditional_entropy - marginal_entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite information-maximization loss.')
        history.append({'step': step + 1, 'im_loss': float(loss.detach())})
    pd.DataFrame(history).to_csv(directory / 'adaptation_history.csv', index=False)


def build_eegnet(C, T, K, fs):
    from braindecode.models import EEGNet
    return EEGNet(n_chans=C, n_outputs=K, n_times=T, sfreq=fs)


_PRETRAINED_FILES = {}


def pinned_converted_labram(frozen, frozen_path):
    """Load the converted LaBraM source recorded by checkpoint metadata."""
    cache = Path(torch.hub.get_dir()) / 'checkpoints'
    cache.mkdir(parents=True, exist_ok=True)
    checkpoint = cache / 'braindecode_labram_base.pt'
    if not checkpoint.exists():
        torch.hub.download_url_to_file(LABRAM_URL, str(checkpoint), progress=True)
    signature = {'repo': LABRAM_URL, 'revision': None,
                 'files': [{'filename': checkpoint.name, 'sha256': file_sha256(checkpoint)}]}
    if frozen != signature:
        raise RuntimeError('LaBraM checkpoint bytes do not match the recorded pretrained identity.')
    atomic_json(frozen_path, signature)
    result = {'repo': LABRAM_URL, 'revision': None, 'config_path': None, 'paths': [str(checkpoint)]}
    _PRETRAINED_FILES['LaBraM'] = result
    return result


def pretrained_files(name):
    """Download official config/checkpoint and record the exact Hub revision."""
    if name in _PRETRAINED_FILES:
        return _PRETRAINED_FILES[name]
    from huggingface_hub import HfApi, hf_hub_download
    repo = FM_REPOS[name][1]
    token = os.environ.get('HF_TOKEN') or None
    frozen_path = Path(CFG.output_root) / 'audit' / f'pretrained_{name}.json'
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else None
    try:
        if name == 'LaBraM' and frozen and frozen['repo'] == LABRAM_URL:
            return pinned_converted_labram(frozen, frozen_path)
        info = HfApi().model_info(repo, token=token,
                                  revision=frozen['revision'] if frozen and frozen['revision'] else None)
        revision = info.sha
        available = {entry.rfilename for entry in info.siblings}
        config_path = hf_hub_download(repo, 'config.json', revision=revision, token=token)
        filenames = []
        for candidate in ('model.safetensors', 'pytorch_model.bin', 'braindecode_labram_base.pt'):
            if candidate in available:
                filenames = [candidate]
                break
        if not filenames and 'model.safetensors.index.json' in available:
            index_path = hf_hub_download(repo, 'model.safetensors.index.json', revision=revision, token=token)
            filenames = sorted(set(json.loads(Path(index_path).read_text())['weight_map'].values()))
        if not filenames:
            candidates = sorted(p for p in available if p.endswith(('.safetensors', '.pt', '.pth', '.bin')))
            if len(candidates) == 1:
                filenames = candidates
            else:
                raise RuntimeError(f'Cannot identify one pretrained checkpoint in {repo}: {candidates}')
        paths = [hf_hub_download(repo, f, revision=revision, token=token) for f in filenames]
        result = {'repo': repo, 'revision': revision, 'config_path': config_path, 'paths': paths}
    except Exception as exc:
        if name != 'LaBraM' or (frozen and frozen['revision']):
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            extra = ('The server returned an access error; check repository permissions and any existing HF_TOKEN.'
                     if status in (401, 403) else 'Check the download error and server connection to Hugging Face.')
            raise RuntimeError(f'{name} pretrained download failed: {exc}. {extra}') from exc
        # Alternate source for the converted LaBraM checkpoint.
        cache = Path(torch.hub.get_dir()) / 'checkpoints'
        cache.mkdir(parents=True, exist_ok=True)
        checkpoint = cache / 'braindecode_labram_base.pt'
        if not checkpoint.exists():
            torch.hub.download_url_to_file(LABRAM_URL, str(checkpoint), progress=True)
        result = {'repo': LABRAM_URL, 'revision': None, 'config_path': None, 'paths': [str(checkpoint)]}
    signature = {'repo': result['repo'], 'revision': result['revision'],
                 'files': [{'filename': Path(p).name, 'sha256': file_sha256(p)} for p in result['paths']]}
    if frozen and frozen != signature:
        raise RuntimeError(f'{name}: pretrained weights differ from the frozen checkpoint identity.')
    atomic_json(frozen_path, signature)
    _PRETRAINED_FILES[name] = result
    return result


def build_foundation(name, C, T, K, channel_names):
    """Load backbone tensors explicitly; initialize the new task head separately."""
    import braindecode.models as models
    files = pretrained_files(name)
    cls = getattr(models, FM_REPOS[name][0])
    config = json.loads(Path(files['config_path']).read_text()) if files['config_path'] else {}
    allowed = set(inspect.signature(cls.__init__).parameters)
    config = {k: v for k, v in config.items() if k in allowed}
    config.update(n_chans=C, n_outputs=K, n_times=T, sfreq=FM_SFREQ,
                  chs_info=[{'ch_name': n} for n in channel_names], input_window_seconds=None)
    if 'return_encoder_output' in allowed:
        config['return_encoder_output'] = False
    model = cls.from_config(config)
    if name == 'REVE':
        bank_names = {n.casefold(): n for n in model._position_bank.get_all_positions()}
        if any(n.casefold() not in bank_names for n in channel_names):
            raise ValueError('REVE position bank does not cover the requested electrodes.')
        model.default_pos = model.get_positions([bank_names[n.casefold()] for n in channel_names])
    source = {}
    identities = []
    for path in files['paths']:
        identities.append({'filename': Path(path).name, 'sha256': file_sha256(path)})
        if str(path).endswith('.safetensors'):
            from safetensors.torch import load_file
            state = load_file(path, device='cpu')
        else:
            state = torch.load(path, map_location='cpu', weights_only=True)
            for key in ('model', 'state_dict'):
                if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                    state = state[key]
        source.update({k: v for k, v in state.items() if torch.is_tensor(v)})
    target = model.state_dict()
    mapping = getattr(model, 'mapping', None) or {}
    variants = []
    for prefix in ('', 'module.', 'model.', 'backbone.'):
        candidate = {mapping.get(k.removeprefix(prefix), k.removeprefix(prefix)): v for k, v in source.items()}
        variants.append(candidate)
    source = max(variants, key=lambda src: sum(v.numel() for k, v in src.items()
                   if k in target and target[k].shape == v.shape))
    def task_head(key):
        return key.startswith(('final_layer.', 'head.', 'classifier.', 'fc_norm.'))
    loaded = {k: v for k, v in source.items() if k in target
              and target[k].shape == v.shape and not task_head(k)}
    backbone = {k: v for k, v in target.items() if not task_head(k)}
    coverage = sum(target[k].numel() for k in loaded) / max(1, sum(v.numel() for v in backbone.values()))
    if coverage < 0.99:
        missing = sorted(set(backbone) - set(loaded))
        raise RuntimeError(f'{name}: only {coverage:.2%} backbone coverage; missing keys {missing[:12]}. No pretrained result will be reported.')
    model.load_state_dict(loaded, strict=False)
    # Always fine-tune the full loaded backbone and a newly initialized classifier.
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    body = [p for n, p in model.named_parameters() if n in loaded]
    head = [p for n, p in model.named_parameters() if n not in loaded]
    groups = [{'params': body, 'lr': REB.fm_lr_backbone}, {'params': head, 'lr': REB.fm_lr_head}]
    kwargs = {'ch_names': channel_names} if name == 'LaBraM' else None
    provenance = {'pretrained_repo': files['repo'], 'pretrained_revision': files['revision'],
                  'pretrained_files': identities, 'backbone_weight_coverage': coverage,
                  'new_or_unloaded_tensors': sorted(set(target) - set(loaded)),
                  'fine_tune_backbone': True}
    return model, groups, kwargs, provenance


def prepare_context(d, fold, cfg):
    train, val, test = training_indices(d, fold, cfg)
    active = np.sort(np.concatenate((train, val, test)))
    local = {n: np.flatnonzero(np.isin(active, idx))
             for n, idx in zip(('train', 'val', 'test'), (train, val, test))}
    X = np.asarray(d.X[active], dtype=np.float32).copy()
    pipeline = ViewPipeline(cfg, d.name).fit(X[local['train']], d.y[train], d.meta.iloc[train].epoch_id.values)
    subjects = d.subject[active].astype(str)
    names = sorted(np.unique(subjects))
    codes = np.array([names.index(s) for s in subjects], dtype=np.int64)
    for key, index in local.items():
        if set(np.unique(d.y[active][index])) != {0, 1}:
            raise ValueError(f'{d.name}/{key} has a missing class.')
    return {'local': local, 'raw': pipeline.raw_transform(X), 'unscaled': X,
            'views': pipeline.transform(X), 'y': d.y[active], 'subject': subjects,
            'domain_codes': codes, 'n_domains': len(names), 'n_classes': 2,
            'channels': list(d.channels), 'meta': d.meta.iloc[active].reset_index(drop=True),
            'active_hash': hashlib.sha256(X.tobytes()).hexdigest(),
            'view_dimensions': pipeline.dimension_rows}


def select_one_fold(d, cfg, root):
    """Reproduce fold 0 from assignment_registry, seed 42, independent of models."""
    subjects = sorted(np.unique(d.subject))
    if d.name == 'EEGET':
        als = sorted(np.unique(d.subject[d.y == 1]))
        controls = sorted(np.unique(d.subject[d.y == 0]))
        if len(als) != 6 or len(controls) < 8:
            raise ValueError('EEGET requires six ALS participants and at least eight controls.')
        paired = np.random.default_rng(cfg.registry_seed).choice(controls, size=6, replace=False)
        tests = (str(als[0]), str(paired[0]))
    else:
        if len(subjects) != {'BNCI': 10, 'UCL': 8}[d.name]:
            raise ValueError(f'{d.name}: unexpected number of participants.')
        tests = (str(subjects[0]),)
    remaining = [s for s in subjects if s not in tests]
    rng = np.random.default_rng(cfg.registry_seed + int(digest([d.name, '0'])[:8], 16))
    if d.name == 'EEGET':
        a = [s for s in remaining if np.all(d.y[d.subject == s] == 1)]
        c = [s for s in remaining if np.all(d.y[d.subject == s] == 0)]
        val = (str(rng.choice(a)), str(rng.choice(c)))
    else:
        val = tuple(sorted(str(s) for s in rng.choice(remaining, cfg.validation_subjects, replace=False)))
    fold = Fold(d.name, '0', tuple(str(s) for s in remaining if s not in val), val, tests,
                'assignment_registry_from_actual_ids; test controls selected before model fitting')
    fold.indices(d)
    path = root / 'audit' / f'{d.name}_fold0.json'
    payload = json.loads(canonical_json(asdict(fold)))
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError(f'{d.name}: frozen fold changed. Use a separate output directory.')
    atomic_json(path, payload)
    return fold


def train_torch(model, X, ctx, spec, seed, device, directory, identity):
    train, val = ctx['local']['train'], ctx['local']['val']
    y, domains = ctx['y'], ctx['domain_codes']
    bs = spec.get('batch_size') or REB.batch_size or CFG.batch_size
    sampler = SubjectClassSampler(y[train], domains[train], bs, seed, REB.per_group)
    loader = torch.utils.data.DataLoader(_DS(_subset(X, train), y[train], domains[train]), batch_sampler=sampler)
    weights = class_weights(y[train], 2).to(device)
    model = model.to(device)
    groups = spec.get('param_groups') or [{'params': [p for p in model.parameters() if p.requires_grad]}]
    optimizer = torch.optim.AdamW(groups, lr=REB.lr, weight_decay=spec.get('weight_decay', REB.weight_decay))
    patience, epochs = spec.get('patience', REB.patience), spec.get('epochs', REB.epochs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=max(3, patience // 3))
    X_val = _subset(X, val)
    kwargs = spec.get('forward_kwargs')
    checkpoint = directory / 'training_state.pt'
    best, best_state, best_epoch, stale, history, start = None, None, -1, 0, [], 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['identity'] != identity:
            raise RuntimeError('Interrupted-job configuration changed; select a separate output directory.')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        best, best_state, best_epoch, stale, history, start = (
            saved[k] for k in ('best', 'best_state', 'best_epoch', 'stale', 'history', 'next_epoch'))
        torch.set_rng_state(saved['torch_rng'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
        print(f'  Resuming training at epoch {start + 1}', flush=True)
    stats = {'n_sk': [], 'valid_groups': [], 'subjects_per_batch': []}
    last_progress = time.monotonic()
    for epoch in range(start, epochs):
        if stale >= patience:
            break
        sampler.epoch = epoch
        model.train()
        if isinstance(model, TSMNetSPDDSMBN):
            model.bn.set_epoch(epoch, epochs, bs)
        losses = []
        for batch, (xb, yb, db) in enumerate(loader):
            xb, yb, db = _to(xb, device), yb.to(device), db.to(device)
            micro = spec.get('micro_batch_size', len(yb))
            optimizer.zero_grad(set_to_none=True)
            value = 0.0
            for offset in range(0, len(yb), micro):
                index = slice(offset, offset + micro)
                if isinstance(model, TSMNetSPDDSMBN):
                    model.current_domains = db[index]
                inputs = {k: v[index] for k, v in xb.items()} if isinstance(xb, dict) else xb[index]
                output = _out(model(inputs, **(kwargs or {})))
                loss = (focal_loss(output['logits'], yb[index], weights)
                        if spec.get('loss') == 'focal' else F.cross_entropy(output['logits'], yb[index], weight=weights))
                if spec.get('lambda_cor', 0):
                    loss = loss + spec['lambda_cor'] * conditional_moment_loss(output['hq'], yb[index], db[index], stats if epoch == 0 else None)
                if spec.get('lambda_sep', 0):
                    loss = loss + spec['lambda_sep'] * separation_loss_eq7(output['hq'], yb[index])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite training loss in epoch {epoch + 1}.')
                fraction = (len(yb[index]) / len(yb) if spec.get('loss') == 'focal'
                            else float(weights[yb[index]].sum() / weights[yb].sum()))
                (loss * fraction).backward()
                value += float(loss.detach()) * fraction
            parameters = [p for group in groups for p in group['params']]
            torch.nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(value)
            if time.monotonic() - last_progress > 40:
                print(f'  epoch {epoch + 1}/{epochs}, batch {batch + 1}/{len(loader)}, loss {value:.4f}', flush=True)
                last_progress = time.monotonic()
        if isinstance(model, TSMNetSPDDSMBN):
            model.current_domains = None
            model.prepare_eval_domain(X_val, device, domains[val])
        pv = predict_probs(model, X_val, device, REB.predict_batch_size, kwargs, domains[val])
        key, threshold = val_criterion(y[val], pv)
        history.append({'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'val_macro_f1': key[0], 'val_auroc': key[1]})
        scheduler.step(key[0])
        if best is None or key > best:
            best, best_epoch, stale = key, epoch + 1, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        atomic_torch(checkpoint, {'identity': identity, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'best': best, 'best_state': best_state, 'best_epoch': best_epoch, 'stale': stale,
            'history': history, 'next_epoch': epoch + 1, 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'numpy_rng': np.random.get_state(), 'python_rng': random.getstate()})
        pd.DataFrame(history).to_csv(directory / 'history.csv', index=False)
        if epoch == start or (epoch + 1) % 5 == 0 or stale >= patience:
            print(f'  epoch {epoch + 1}: source-val Macro-F1 {key[0]:.4f}, best epoch {best_epoch}', flush=True)
            last_progress = time.monotonic()
    if best_state is None:
        raise RuntimeError('Training produced no validated checkpoint.')
    model.load_state_dict(best_state)
    if isinstance(model, TSMNetSPDDSMBN):
        model.prepare_eval_domain(X_val, device, domains[val])
    pv = predict_probs(model, X_val, device, REB.predict_batch_size, kwargs, domains[val])
    threshold = select_threshold_val(y[val], pv)
    return model, {'threshold': threshold, 'best_epoch': best_epoch,
                   'batch_stats': {k: float(np.median(v)) if v else None for k, v in stats.items()}}, pv


def _fit_ea_ts_lr_unregularized(X, ctx, seed):
    train, val = ctx['local']['train'], ctx['local']['val']
    best = None
    for C in (0.01, 0.1, 1.0, 10.0):
        model = Pipeline([('cov', Covariances('oas')), ('ts', TangentSpace(metric='riemann')),
                          ('scale', StandardScaler()), ('clf', LogisticRegression(
                              C=C, class_weight='balanced', max_iter=5000, random_state=seed))])
        model.fit(X[train].astype(np.float64), ctx['y'][train])
        probability = model.predict_proba(X[val].astype(np.float64))
        key, threshold = val_criterion(ctx['y'][val], probability)
        if best is None or key > best[0]:
            best = key, model, threshold, probability
    return best[1], {'threshold': best[2], 'best_epoch': None}, best[3]

from sklearn.base import BaseEstimator, TransformerMixin


EA_TS_LR_NUMERICAL_POLICY = 'ea_ts_lr_spd_guard_v1'
_EA_TS_LR_NUMERICS = {}


class StrictSPDCovariances(TransformerMixin, BaseEstimator):
    """Fixed eigenvalue floor; scale is fitted on source-training covariances."""

    def __init__(self, relative_floor=1e-6):
        self.relative_floor = relative_floor

    @staticmethod
    def _batch(X):
        values = np.asarray(X, dtype=np.float64)
        if (values.ndim != 3 or values.shape[1] != values.shape[2]
                or not len(values) or not np.isfinite(values).all()):
            raise ValueError('EA + TS-LR requires finite square covariance matrices.')
        return (values + values.transpose(0, 2, 1)) * 0.5

    def fit(self, X, y=None):
        values = self._batch(X)
        if not 0 < self.relative_floor < 1:
            raise ValueError('The covariance relative floor must be between zero and one.')
        scales = np.trace(values, axis1=1, axis2=2) / values.shape[1]
        positive = scales[scales > 0]
        self.source_scale_ = float(np.median(positive)) if len(positive) else 1.0
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        if not hasattr(self, 'source_scale_'):
            raise RuntimeError('Fit the covariance floor on source training data first.')
        values = self._batch(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError('The covariance channel count changed.')
        eigenvalues, eigenvectors = np.linalg.eigh(values)
        scale = np.maximum(eigenvalues[:, -1], self.source_scale_)
        floor = self.relative_floor * scale
        eigenvalues = np.maximum(eigenvalues, floor[:, None])
        result = (eigenvectors * eigenvalues[:, None, :]) @ eigenvectors.transpose(0, 2, 1)
        result = (result + result.transpose(0, 2, 1)) * 0.5
        if not np.isfinite(result).all() or np.any(np.linalg.eigvalsh(result) <= 0):
            raise FloatingPointError('EA + TS-LR covariance regularization did not produce finite SPD matrices.')
        return result


def _ea_ts_lr_check_probability(probability, count):
    probability = np.asarray(probability)
    if (probability.shape != (count, 2) or not np.isfinite(probability).all()
            or np.any(probability < 0) or np.any(probability > 1)
            or not np.allclose(probability.sum(axis=1), 1, atol=1e-5)):
        raise FloatingPointError('Non-finite or invalid EA + TS-LR tangent-space probabilities.')


def _ea_ts_lr_numerical_error(error):
    if isinstance(error, (np.linalg.LinAlgError, FloatingPointError, RuntimeWarning)):
        return True
    message = str(error).lower()
    return any(text in message for text in (
        'positive definite', 'singular matrix', 'eigenvalues did not converge',
        'input x contains nan', 'input contains nan', 'infs or nans',
        'infinity or a value too large'))


def fit_ea_ts_lr(X, ctx, seed):
    """Fit OAS/Riemannian TS-LR with a conditional numerical covariance floor."""
    if not np.isfinite(X).all():
        raise ValueError('EA + TS-LR input EEG contains NaN or infinity; inspect preprocessing.')
    train, val, test = (ctx['local'][part] for part in ('train', 'val', 'test'))
    trigger = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=RuntimeWarning, module=r'pyriemann\..*')
            model, artifact, probability = _fit_ea_ts_lr_unregularized(X, ctx, seed)
            _ea_ts_lr_check_probability(probability, len(val))
            # No held-out labels are read here. This checks numerical validity
            # of the same transform that run_model subsequently uses for test.
            _ea_ts_lr_check_probability(model.predict_proba(X[test].astype(np.float64)), len(test))
    except (ValueError, np.linalg.LinAlgError, FloatingPointError, RuntimeWarning) as error:
        if not _ea_ts_lr_numerical_error(error):
            raise
        trigger = f'{type(error).__name__}: {error}'
        print('  EA + TS-LR: applying the fixed covariance eigenvalue floor (1e-6).', flush=True)
        best = None
        for C in (0.01, 0.1, 1.0, 10.0):
            model = Pipeline([
                ('cov', Covariances('oas')),
                ('spd', StrictSPDCovariances(relative_floor=1e-6)),
                ('ts', TangentSpace(metric='riemann')),
                ('scale', StandardScaler()),
                ('clf', LogisticRegression(C=C, class_weight='balanced', max_iter=5000,
                                           random_state=seed)),
            ])
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=RuntimeWarning, module=r'pyriemann\..*')
                model.fit(X[train].astype(np.float64), ctx['y'][train])
                probability = model.predict_proba(X[val].astype(np.float64))
                _ea_ts_lr_check_probability(probability, len(val))
            key, threshold = val_criterion(ctx['y'][val], probability)
            if best is None or key > best[0]:
                best = key, model, threshold, probability
        model, probability = best[1], best[3]
        artifact = {'threshold': best[2], 'best_epoch': None}
        _ea_ts_lr_check_probability(model.predict_proba(X[test].astype(np.float64)), len(test))
    numerical = {'policy': EA_TS_LR_NUMERICAL_POLICY, 'applied': trigger is not None,
                 'relative_floor': 1e-6 if trigger is not None else None,
                 'scale_fit': 'source training covariance only' if trigger is not None else None,
                 'source_scale': model.named_steps['spd'].source_scale_ if trigger is not None else None,
                 'trigger': trigger, 'selected_C': float(model.named_steps['clf'].C)}
    model.ea_ts_lr_numerics_ = numerical
    artifact['ea_ts_lr_numerics'] = numerical
    _EA_TS_LR_NUMERICS[int(seed)] = numerical
    return model, artifact, probability



UCL_ELIGIBILITY_POLICY = 'ucl_minimum_resampled_length_v1'
def run_identity(name, ctx, fold, seed):
    weights = None
    if name in FOUNDATION:
        path = Path(CFG.output_root) / 'audit' / f'pretrained_{name}.json'
        if not path.exists():
            pretrained_files(name)
        weights = json.loads(path.read_text())
    return digest({'code': SOURCE_CODE_SHA256, 'config': asdict(CFG), 'training': asdict(REB),
        'pretrained': weights,
        'fold': asdict(fold), 'seed': seed, 'model': name, 'data': ctx['active_hash'],
        'labels': ctx['y'], 'trials': ctx['meta'].epoch_id.tolist(),
        'versions': {p: package_metadata.version(p) for p in
                     ('torch', 'braindecode', 'pennylane', 'numpy', 'scipy', 'scikit-learn', 'mne', 'pyriemann')}})


def run_model(name, data, fold, ctx, seed, device, root):
    if name not in MODEL_NAMES:
        raise ValueError('Model outside the configured experiment scope: ' + name)
    directory = root / data.name / 'fold0' / name / f'seed_{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    identity = run_identity(name, ctx, fold, seed)
    result_path, predictions_path = directory / 'result.json', directory / 'predictions.npz'
    test, train, val = (ctx['local'][k] for k in ('test', 'train', 'val'))
    if result_path.exists() and predictions_path.exists():
        record = json.loads(result_path.read_text())
        if record.get('identity') != identity:
            raise RuntimeError('Completed-job configuration differs. Choose a separate output directory.')
        with np.load(predictions_path, allow_pickle=False) as saved:
            if (not np.array_equal(saved['epoch_id'], ctx['meta'].iloc[test].epoch_id.values.astype(str))
                    or not np.array_equal(saved['y_true'], ctx['y'][test])):
                raise ValueError('Cached predictions do not match the current held-out trials.')
            measured = metrics_from_probs(saved['y_true'], saved['prob'], float(saved['threshold']))
        if any(abs(measured[k] - record[k]) > 1e-9 for k in ('macro_f1', 'bacc', 'auroc', 'auprc')):
            raise ValueError('Cached metrics do not match saved predictions.')
        record['resumed_from_disk'] = True
        return record
    seed_everything(seed)
    started = time.perf_counter()
    C, T = ctx['raw'].shape[1:]
    info, spec, model = {}, {'loss': 'ce'}, None
    if name == 'EA_TS_LR':
        X = euclidean_alignment(ctx['raw'], ctx['subject'])
        model, artifact, val_prob = fit_ea_ts_lr(X, ctx, seed)
        probability = model.predict_proba(X[test].astype(np.float64))
        n_parameters = None
        joblib.dump(model, directory / 'model.joblib')
    else:
        if name in VIEW_MODELS:
            X = ctx['views']
            dims = {k: v.shape[1] for k, v in X.items()}
            if name.startswith('Residual_only'):
                model = build_residual_only(dims, CFG, 2, name.endswith('_matched'))
            else:
                model = MultiViewNetwork(dims, CFG, {'encoder': 'VQC', 'residual': False})
            spec.update(loss='focal', lambda_cor=CFG.lambda_cor, lambda_sep=CFG.lambda_sep)
        elif name in FOUNDATION:
            X = fm_inputs(ctx, CFG.fs, name)
            model, groups, kwargs, info = build_foundation(name, C, X.shape[-1], 2,
                                                         canonical_channel_names(ctx['channels']))
            spec.update(param_groups=groups, forward_kwargs=kwargs, epochs=REB.fm_epochs,
                        patience=REB.fm_patience, batch_size=REB.fm_batch_size,
                        micro_batch_size=REB.fm_micro_batch_size, weight_decay=REB.fm_weight_decay)
        elif name == 'EA_EEGNet':
            X = euclidean_alignment(ctx['raw'], ctx['subject'])
            model = build_eegnet(C, T, 2, CFG.fs)
        else:
            X = ctx['raw']
            if name == 'SPDNet':
                model = SPDNet(C, 2)
            elif name == 'MAtt':
                model = MAttStyle(C, T, 2)
            elif name in ('TSMNet_SPDDSMBN', 'SPDIM'):
                model = TSMNetSPDDSMBN(C, T, 2, ctx['n_domains'])
            else:
                raise ValueError(f'Unknown model: {name}')
        n_parameters = n_trainable(model)
        model, artifact, val_prob = train_torch(model, X, ctx, spec, seed, device, directory, identity)
        if isinstance(model, TSMNetSPDDSMBN):
            X_test = _subset(X, test)
            if name == 'SPDIM':
                spdim_adapt(model, X_test, ctx['domain_codes'][test], device,
                            REB.spdim_steps, REB.spdim_lr, directory)
            else:
                model.prepare_eval_domain(X_test, device, ctx['domain_codes'][test])
        probability = predict_probs(model, _subset(X, test), device, REB.predict_batch_size,
                                    spec.get('forward_kwargs'), ctx['domain_codes'][test])
        atomic_torch(directory / 'model.pt', {k: v.detach().cpu() for k, v in model.state_dict().items()})
    measured = metrics_from_probs(ctx['y'][test], probability, artifact['threshold'])
    record = {'identity': identity, 'dataset': data.name, 'fold_id': '0', 'seed': seed,
              'model': name, 'test_subjects': list(fold.test_subjects),
              'train_subjects': list(fold.train_subjects), 'val_subjects': list(fold.val_subjects),
              'n_train': len(train), 'n_val': len(val), 'n_test': len(test),
              'transductive': name in TRANSDUCTIVE, 'n_params': n_parameters,
              'best_epoch': artifact['best_epoch'], 'batch_stats': artifact.get('batch_stats'),
              'seconds': time.perf_counter() - started, 'status': 'success', **info, **measured}
    if name == 'EA_TS_LR':
        record['ea_ts_lr_numerics'] = artifact['ea_ts_lr_numerics']
    np.savez_compressed(directory / 'validation_predictions.npz', y_true=ctx['y'][val],
                        prob=val_prob, epoch_id=ctx['meta'].iloc[val].epoch_id.values.astype(str))
    temporary = directory / 'predictions.tmp.npz'
    np.savez_compressed(temporary, y_true=ctx['y'][test], prob=probability,
        threshold=np.array(artifact['threshold']), epoch_id=ctx['meta'].iloc[test].epoch_id.values.astype(str),
        subject=ctx['subject'][test].astype(str))
    os.replace(temporary, predictions_path)
    atomic_json(result_path, record)
    del model, X
    gc.collect()
    torch.cuda.empty_cache()
    return record


BASELINE_ROWS = [
    ('LaBraM (fine-tuned)', 'LaBraM'), ('CBraMod (fine-tuned)', 'CBraMod'),
    ('REVE (fine-tuned)', 'REVE'), ('MAtt', 'MAtt'), ('SPDNet', 'SPDNet'),
    ('TSMNet (SPDDSMBN)†', 'TSMNet_SPDDSMBN'),
    ('SPDIM-style (IM-adapted SPD bias)†', 'SPDIM'),
    ('EA + EEGNet† / EA + TS-LR†', ('EA_EEGNet', 'EA_TS_LR')),
]
ABLATION_ROWS = [('Residual only', 'Residual_only'), ('Residual only, parameter-matched', 'Residual_only_matched'), ('Quantum only', 'Quantum_only')]


def export_tables(rows, root):
    directory = root / 'results'
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(directory / 'per_seed_results.csv', index=False)
    stats = []
    for dataset in REB.datasets:
        for model in REB.models:
            group = frame[(frame.dataset == dataset) & (frame.model == model) & (frame.status == 'success')]
            complete = set(group.seed.astype(int)) == set(REB.seeds) and len(group) == len(REB.seeds)
            values = group.macro_f1.to_numpy(dtype=float) * 100 if 'macro_f1' in group else np.array([])
            stats.append({'dataset': dataset, 'model': model, 'n_seeds': len(group),
                          'mean_percent': float(values.mean()) if complete else None,
                          'sd_percent': float(values.std(ddof=1)) if complete else None,
                          'complete': complete})
    statistics = pd.DataFrame(stats)
    statistics.to_csv(directory / 'macro_f1_mean_sd.csv', index=False)
    lookup = {(r['dataset'], r['model']): r for r in stats}
    def cell(dataset, model):
        r = lookup[dataset, model]
        return (f"{r['mean_percent']:.2f} ± {r['sd_percent']:.2f}"
                if r['complete'] else f"INCOMPLETE ({r['n_seeds']}/3)")
    notes = ('Each entry is Macro-F1 (%) mean ± sample SD (ddof=1) over seeds 11, 22, 33 on fold 0.\n'
             'BNCI/UCL each test one participant; EEGET tests one ALS participant plus its paired control.\n'
             '† Uses unlabeled held-out trials. All checkpoint and threshold selection uses source validation only.\n'
             'MAtt, SPDNet and TSMNet are reimplementations adapted to these datasets and the common trainer.\n'
             'SPDIM-style adapts only the tangent-space SPD bias by fixed-step information maximization.\n'
             'SD measures variation across seeds on this fold, not variation across participants.\n'
             'UCL uses only trials supplying the fixed two-second window; excluded trial IDs and participant/class counts are recorded in audit/UCL_excluded_short_trials.csv and audit/UCL_eligibility_counts.csv.\n'
             'EA + TS-LR uses a conditional 1e-6 covariance eigenvalue floor with source-training scale. Activation is recorded per evaluation.\n')
    for filename, definitions in [('baseline_table', BASELINE_ROWS), ('ablation_table', ABLATION_ROWS)]:
        table_rows = []
        for label, model in definitions:
            table_rows.append([label] + [(' / '.join(cell(d, m) for m in model)
                                          if isinstance(model, tuple) else cell(d, model)) for d in REB.datasets])
        table = pd.DataFrame(table_rows, columns=['Macro-F1 (%)', *REB.datasets])
        table.to_csv(directory / f'{filename}.csv', index=False)
        markdown = ['| Macro-F1 (%) | ' + ' | '.join(REB.datasets) + ' |',
                    '|---|' + '---|' * len(REB.datasets)]
        markdown += ['| ' + ' | '.join(row) + ' |' for row in table_rows]
        text = '\n'.join(markdown) + '\n\n' + notes
        (directory / f'{filename}.md').write_text(text)
    return frame, statistics


def numerical_self_tests(device):
    """Synthetic checks only: never substitutes synthetic data for experiments."""
    seed_everything(2026)
    torch.set_num_threads(max(1, int(os.environ.get('MVQAN_TORCH_THREADS', '4'))))
    identity = torch.eye(3, dtype=torch.float64, device=device).requires_grad_(True)
    if not torch.autograd.gradcheck(logm, (identity,), eps=1e-6, atol=1e-5, rtol=1e-4):
        raise AssertionError('LogEig repeated-eigenvalue gradient check failed.')
    tests = ['LogEig gradient at repeated eigenvalues']
    for C, T in ((16, 103), (19, 256), (32, 256)):
        X = torch.randn(4, C, T, device=device, requires_grad=True)
        domains = torch.tensor([0, 0, 1, 1], device=device)
        for name, model in [('MAtt', MAttStyle(C, T, 2)), ('SPDNet', SPDNet(C, 2)),
                            ('TSMNet_SPDDSMBN', TSMNetSPDDSMBN(C, T, 2, 2))]:
            model.to(device).train()
            if isinstance(model, TSMNetSPDDSMBN):
                model.current_domains = domains
            output = model(X)['logits']
            assert output.shape == (4, 2) and torch.isfinite(output).all()
            F.cross_entropy(output, torch.tensor([0, 1, 0, 1], device=device)).backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert X.grad is not None and torch.isfinite(X.grad).all() and X.grad.abs().sum() > 0
            X.grad = None
            if isinstance(model, TSMNetSPDDSMBN):
                model.prepare_eval_domain(X.cpu(), device, domains.cpu().numpy())
                predict_probs(model, X.cpu(), device, domains=domains.cpu().numpy())
            tests.append(f'{name}: forward/backward {C} channels, {T} samples')
        del model
    dims = {f'view{i}': i + 4 for i in range(3)}
    views = {k: torch.randn(3, n, device=device) for k, n in dims.items()}
    for label, model in [('residual', build_residual_only(dims, CFG, 2)),
                         ('matched', build_residual_only(dims, CFG, 2, True)),
                         ('quantum', MultiViewNetwork(dims, CFG, {'encoder': 'VQC', 'residual': False}))]:
        model.to(device)
        output = model(views)['logits']
        assert output.shape == (3, 2) and torch.isfinite(output).all()
        output.square().sum().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if label == 'quantum':
            assert all(e.core.weights.grad is not None and e.core.weights.grad.abs().sum() > 0
                       for e in model.encoders.values())
        tests.append(f'{label}: forward/backward')
    for shape in ((512, 19), (19, 512), (3, 19, 512), (512, 22, 3)):
        trials = as_ucl_trials(np.zeros(shape, dtype=np.float32))
        assert all(t.shape == (19, 512) for t in trials)
    tests.append('UCL MATLAB array orientations')
    gc.collect()
    torch.cuda.empty_cache()
    return tests


def stage_pretrained_metadata(source_directory, output_root):
    """Use recorded checkpoint identities as explicit replication inputs."""
    if source_directory is None:
        return
    source_directory = Path(source_directory).expanduser().resolve()
    plans = []
    for model in ('LaBraM', 'CBraMod', 'REVE'):
        source = source_directory / f'pretrained_{model}.json'
        metadata = json.loads(source.read_text())
        if (not isinstance(metadata.get('repo'), str)
                or metadata.get('revision') is not None and not isinstance(metadata['revision'], str)
                or not isinstance(metadata.get('files'), list) or not metadata['files']):
            raise ValueError('Invalid pretrained checkpoint metadata: ' + str(source))
        for entry in metadata['files']:
            sha = entry.get('sha256', '')
            if (not isinstance(entry.get('filename'), str) or len(sha) != 64
                    or any(character not in '0123456789abcdef' for character in sha)):
                raise ValueError('Invalid pretrained checkpoint file identity: ' + str(source))
        destination = output_root / 'audit' / source.name
        if destination.exists() and json.loads(destination.read_text()) != metadata:
            raise ValueError('The output directory already records a different pretrained checkpoint: ' + model)
        plans.append((destination, metadata))
    for destination, metadata in plans:
        atomic_json(destination, metadata)


def main(args=None):
    global CFG
    args = parse_arguments([]) if args is None else args
    CFG = replace(CFG, data_root=str(args.data_root.expanduser().resolve()),
                  output_root=str(args.output_root.expanduser().resolve()))
    CFG.ucl_manifest_csv = str(Path(CFG.data_root) / 'ucl_manifest.csv')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. Use a CUDA-enabled PyTorch environment and CUDA_VISIBLE_DEVICES=0.')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible not in ('0', None):
        raise RuntimeError(f'This run is restricted to physical GPU 0; CUDA_VISIBLE_DEVICES={visible!r}. Restart with 0.')
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    root = Path(CFG.output_root)
    for subdir in ('audit', 'logs', 'results'):
        (root / subdir).mkdir(parents=True, exist_ok=True)
    stage_pretrained_metadata(args.pretrained_metadata, root)
    print('GPU 0:', torch.cuda.get_device_name(0), flush=True)
    print('Data:', CFG.data_root, flush=True)
    print('Output:', root, flush=True)
    print('Plan: 12 configurations × 3 datasets × 3 seeds = 108 jobs; fold 0 only.', flush=True)
    environment = {'python': sys.version, 'gpu': torch.cuda.get_device_name(0),
                   'cuda': torch.version.cuda, 'code_sha256': SOURCE_CODE_SHA256,
                   'packages': {p: package_metadata.version(p) for p in
                     ('numpy', 'scipy', 'pandas', 'torch', 'mne', 'moabb', 'pyriemann',
                      'pennylane', 'braindecode', 'huggingface_hub', 'safetensors')}}
    atomic_json(root / 'audit/environment.json', environment)
    atomic_json(root / 'audit/config.json', {'config': asdict(CFG), 'training': asdict(REB)})
    print('Checking matrix gradients and all requested local architectures ...', flush=True)
    checks = numerical_self_tests(device)
    atomic_json(root / 'audit/self_tests.json', {'passed': checks})
    print(f'Passed {len(checks)} numerical/shape checks.', flush=True)
    if args.self_test_only:
        return checks
    rows, blockers = [], {}
    for dataset in REB.datasets:
        print(f'\nLoading {dataset_display_name(dataset)}', flush=True)
        try:
            loader = {'BNCI': load_bnci, 'UCL': load_ucl, 'EEGET': load_eeget}[dataset]
            with execution_log(CFG, root / 'logs' / f'load_{dataset}.log'):
                data, cache = load_prepared_dataset(dataset, CFG, loader)
            atomic_json(root / 'audit' / f'{dataset}_cache.json', cache)
            atomic_json(root / 'audit' / f'{dataset}_provenance.json', data.provenance)
            data.meta.groupby(['subject', 'y']).size().rename('n').reset_index().to_csv(
                root / 'audit' / f'{dataset}_participant_counts.csv', index=False)
            print(f'  {data.X.shape}; {len(np.unique(data.subject))} participants; {cache["status"]}', flush=True)
            fold = select_one_fold(data, CFG, root)
            print(f'  fold 0: test={fold.test_subjects}, validation={fold.val_subjects}, sources={len(fold.train_subjects)}', flush=True)
            with execution_log(CFG, root / 'logs' / f'views_{dataset}.log'):
                ctx = prepare_context(data, fold, CFG)
            atomic_json(root / 'audit' / f'{dataset}_views.json', ctx['view_dimensions'])
        except Exception as exc:
            blockers[dataset] = f'{type(exc).__name__}: {exc}'
            (root / 'logs' / f'{dataset}_dataset_error.log').write_text(traceback.format_exc())
            print(f'  DATASET BLOCKED: {exc}', flush=True)
            rows.extend({'dataset': dataset, 'fold_id': '0', 'model': model, 'seed': seed,
                         'status': 'blocked', 'error': blockers[dataset]} for model in REB.models for seed in REB.seeds)
            export_tables(rows, root)
            continue
        for model in REB.models:
            for seed in REB.seeds:
                display_model = {internal: label for label, internal in ABLATION_ROWS}.get(model, model)
                print(f'\n[{len(rows) + 1}/108] {dataset} {display_model} seed {seed}', flush=True)
                try:
                    record = run_model(model, data, fold, ctx, seed, device, root)
                    print(f'  Macro-F1={100 * record["macro_f1"]:.2f}%'
                          + (' (verified saved result)' if record.get('resumed_from_disk') else ''), flush=True)
                except Exception as exc:
                    record = {'dataset': dataset, 'fold_id': '0', 'model': model, 'seed': seed,
                              'status': 'blocked', 'error': f'{type(exc).__name__}: {exc}'}
                    (root / 'logs' / f'{dataset}_{model}_seed{seed}_error.log').write_text(traceback.format_exc())
                    print(f'  BLOCKED: {exc}', flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                rows.append(record)
                export_tables(rows, root)
        del ctx, data
        gc.collect()
    frame, statistics = export_tables(rows, root)
    complete = len(frame) == 108 and bool((frame.status == 'success').all())
    failures = frame[frame.status != 'success'][['dataset', 'model', 'seed', 'error']].to_dict('records') if not complete else []
    atomic_json(root / 'results/completion.json', {'complete': complete, 'expected': 108,
                'successful': int((frame.status == 'success').sum()), 'attempted_or_blocked': len(frame),
                'data_blockers': blockers, 'failed_jobs': failures, 'seed_sd_ddof': 1})
    print('\n' + (root / 'results/baseline_table.md').read_text(), flush=True)
    print((root / 'results/ablation_table.md').read_text(), flush=True)
    if not complete:
        raise RuntimeError(f'Incomplete: {int((frame.status == "success").sum())}/108 successful. '
                           f'See {root / "results/completion.json"}. See the corresponding error logs; matching completed jobs are reused on restart.')
    print('COMPLETE: 108/108 jobs. Rebuttal tables:', root / 'results', flush=True)
    return frame




if __name__ == '__main__':
    main(ARGS)
