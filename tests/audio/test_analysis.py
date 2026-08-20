"""Diagnostics: RMS, per-channel energy, spectral features, WAV round-trip."""

import numpy as np
import pytest

from heimdall.audio import synthetic
from heimdall.audio.analysis import (
    analyse_frame,
    band_energy_ratio,
    channel_peak,
    channel_rms,
    concatenate_frames,
    db_fs,
    peak,
    read_wav,
    rms,
    spectral_flatness,
    spectrogram,
    write_wav,
    zero_crossing_rate,
)
from heimdall.audio.frame import AudioFrame

SAMPLE_RATE = 48000


def frame_from(samples, index=0):
    samples = np.asarray(samples, dtype=np.float32)
    return AudioFrame(samples, index * samples.shape[0] / SAMPLE_RATE, index, SAMPLE_RATE)


def test_rms_of_silence_is_zero():
    assert rms(np.zeros(1024)) == 0.0


def test_rms_of_a_constant_is_its_magnitude():
    assert rms(np.full(512, 0.5)) == pytest.approx(0.5)


def test_rms_of_a_sine_is_amplitude_over_root_two():
    x = synthetic.tone(48000, SAMPLE_RATE, frequency=1000.0, amplitude=1.0)
    assert rms(x) == pytest.approx(1 / np.sqrt(2), rel=1e-3)


def test_rms_of_empty_input_is_zero_not_nan():
    assert rms(np.array([])) == 0.0
    assert peak(np.array([])) == 0.0


def test_peak_finds_the_largest_magnitude():
    assert peak(np.array([0.1, -0.9, 0.4])) == pytest.approx(0.9)


def test_per_channel_energy_is_independent():
    samples = np.zeros((1024, 2), dtype=np.float32)
    samples[:, 0] = 0.5
    samples[:, 1] = 0.1
    levels = channel_rms(frame_from(samples))
    assert levels.shape == (2,)
    assert levels[0] == pytest.approx(0.5)
    assert levels[1] == pytest.approx(0.1)


def test_per_channel_peak():
    samples = np.zeros((256, 2), dtype=np.float32)
    samples[10, 0] = 0.8
    samples[20, 1] = -0.3
    peaks = channel_peak(frame_from(samples))
    assert peaks[0] == pytest.approx(0.8)
    assert peaks[1] == pytest.approx(0.3)


def test_per_channel_energy_works_for_four_channels():
    samples = np.tile(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32), (512, 1))
    levels = channel_rms(frame_from(samples))
    assert levels.shape == (4,)
    assert np.allclose(levels, [0.1, 0.2, 0.3, 0.4])


def test_db_fs_of_full_scale_is_zero():
    assert db_fs(1.0) == pytest.approx(0.0)
    assert db_fs(0.5) == pytest.approx(-6.02, abs=0.05)


def test_db_fs_of_zero_is_very_negative_not_infinite():
    assert np.isfinite(db_fs(0.0))
    assert db_fs(0.0) < -200


def test_zero_crossing_rate_bounds():
    assert zero_crossing_rate(np.ones(100)) == 0.0
    alternating = np.array([1.0, -1.0] * 50)
    assert zero_crossing_rate(alternating) == pytest.approx(1.0)


def test_spectral_flatness_separates_tone_from_noise():
    tone_flatness = spectral_flatness(synthetic.tone(4096, SAMPLE_RATE, 1000.0))
    noise_flatness = spectral_flatness(synthetic.white_noise(4096, seed=1))
    assert tone_flatness < 0.1
    assert noise_flatness > tone_flatness


def test_band_energy_ratio_captures_an_in_band_tone():
    in_band = band_energy_ratio(synthetic.tone(8192, SAMPLE_RATE, 1000.0), SAMPLE_RATE)
    out_of_band = band_energy_ratio(synthetic.tone(8192, SAMPLE_RATE, 12000.0), SAMPLE_RATE)
    assert in_band > 0.9
    assert out_of_band < 0.1


def test_analyse_frame_reports_every_field():
    mono = synthetic.speech_like(2048, SAMPLE_RATE, seed=1)
    samples = np.stack([mono, mono], axis=1)
    stats = analyse_frame(frame_from(samples, index=3))

    assert stats.frame_index == 3
    assert stats.rms > 0
    assert stats.peak >= stats.rms
    assert len(stats.channel_rms) == 2
    assert len(stats.channel_peak) == 2
    assert 0.0 <= stats.speech_band_ratio <= 1.0
    assert 0.0 <= stats.spectral_flatness <= 1.0
    assert stats.rms_dbfs < 0


def test_spectrogram_shape_and_frequency_axis():
    x = synthetic.tone(8192, SAMPLE_RATE, 1000.0)
    freqs, times, magnitude = spectrogram(x, SAMPLE_RATE, window_size=512, hop=256)
    assert magnitude.shape[0] == freqs.size == 257
    assert magnitude.shape[1] == times.size
    assert freqs[-1] == pytest.approx(SAMPLE_RATE / 2)


def test_spectrogram_finds_the_tone():
    x = synthetic.tone(8192, SAMPLE_RATE, 2000.0)
    freqs, _, magnitude = spectrogram(x, SAMPLE_RATE, window_size=1024, hop=512)
    dominant = freqs[np.argmax(magnitude.mean(axis=1))]
    assert dominant == pytest.approx(2000.0, abs=SAMPLE_RATE / 1024)


def test_spectrogram_handles_input_shorter_than_the_window():
    freqs, times, magnitude = spectrogram(np.zeros(100), SAMPLE_RATE, window_size=512)
    assert magnitude.shape[1] >= 1


def test_wav_round_trip_preserves_samples(tmp_path):
    original = np.stack(
        [
            synthetic.speech_like(4096, SAMPLE_RATE, seed=1),
            synthetic.speech_like(4096, SAMPLE_RATE, seed=2),
        ],
        axis=1,
    ).astype(np.float32)

    path = write_wav(tmp_path / "two_channel.wav", original, SAMPLE_RATE)
    loaded, sample_rate = read_wav(path)

    assert sample_rate == SAMPLE_RATE
    assert loaded.shape == original.shape
    # 16-bit quantisation is the only loss.
    assert np.max(np.abs(loaded - original)) < 2e-4


def test_wav_keeps_channels_separate(tmp_path):
    samples = np.zeros((1024, 2), dtype=np.float32)
    samples[:, 0] = 0.5
    samples[:, 1] = -0.25

    path = write_wav(tmp_path / "channels.wav", samples, SAMPLE_RATE)
    loaded, _ = read_wav(path)

    assert loaded[:, 0].mean() == pytest.approx(0.5, abs=1e-4)
    assert loaded[:, 1].mean() == pytest.approx(-0.25, abs=1e-4)


def test_wav_clips_rather_than_wrapping(tmp_path):
    samples = np.full((256, 1), 2.0, dtype=np.float32)
    path = write_wav(tmp_path / "clip.wav", samples, SAMPLE_RATE)
    loaded, _ = read_wav(path)
    assert np.all(loaded > 0.99)


def test_wav_sample_rate_is_configurable(tmp_path):
    for sample_rate in (16000, 44100, 48000):
        path = write_wav(tmp_path / f"sr_{sample_rate}.wav", np.zeros((64, 2)), sample_rate)
        _, loaded_rate = read_wav(path)
        assert loaded_rate == sample_rate


def test_concatenate_frames_stacks_in_order():
    frames = [frame_from(np.full((128, 2), i, dtype=np.float32), index=i) for i in range(3)]
    stacked = concatenate_frames(frames)
    assert stacked.shape == (384, 2)
    assert stacked[0, 0] == 0 and stacked[128, 0] == 1 and stacked[256, 0] == 2


def test_concatenate_rejects_mismatched_channel_counts():
    frames = [frame_from(np.zeros((16, 2), np.float32)), frame_from(np.zeros((16, 4), np.float32))]
    with pytest.raises(ValueError):
        concatenate_frames(frames)


def test_concatenate_empty_list_is_safe():
    assert concatenate_frames([]).size == 0
