import argparse
import time
import numpy as np
import sounddevice as sd
from scipy.signal import resample

from tflite_support.task import audio
from tflite_support.task import core
from tflite_support.task import processor
from ai_agent.mqtt.client import MQTTPublisher

mqtt = MQTTPublisher()

def list_input_devices():
    print("🔍 Listing ALSA audio capture devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            marker = ">" if i == sd.default.device[0] else " "
            print(f"{marker} {i} {dev['name']} ({dev['hostapi']})")

def run(model: str, max_results: int, score_threshold: float,
        overlapping_factor: float, num_threads: int,
        enable_edgetpu: bool) -> None:
    """Continuously run inference on audio data from microphone."""

    # Model setup
    base_options = core.BaseOptions(
        file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
    classification_options = processor.ClassificationOptions(
        max_results=max_results, score_threshold=score_threshold)
    options = audio.AudioClassifierOptions(
        base_options=base_options, classification_options=classification_options)
    classifier = audio.AudioClassifier.create_from_options(options)

    # Audio config
    model_sample_rate = classifier.required_audio_format.sample_rate
    duration = 1.0  # seconds
    mic_sample_rate = 44100  # standard mic sample rate

    tensor_audio = classifier.create_input_tensor_audio()
    input_buffer_size = len(tensor_audio.buffer)
    print(f"[INFO] Mic sample rate: {mic_sample_rate}, Model expects: {model_sample_rate}")
    print(f"[INFO] Input buffer size: {input_buffer_size}, Duration: {duration} sec")

    audio_buffer = np.zeros(int(mic_sample_rate * duration), dtype=np.float32)

    def audio_callback(indata, frames, time_info, status):
        nonlocal audio_buffer
        if status:
            print(f"[WARN] Sounddevice status: {status}")
        try:
            print(f"[DEBUG] Raw input shape: {indata.shape}")

            # Robust mono conversion
            if indata.ndim == 2:
                mono_data = indata.mean(axis=1)
            else:
                mono_data = indata

            mono_data = mono_data.astype(np.float32).flatten()
            print(f"[DEBUG] Mono min: {mono_data.min()}, max: {mono_data.max()}")
            print(f"[DEBUG] mono_data len: {len(mono_data)}, audio_buffer len: {len(audio_buffer)}")
            if len(mono_data) == 0:
                print("[ERROR] mono_data is empty!")
                return
            if len(mono_data) > len(audio_buffer):
                print("[WARN] mono_data larger than audio_buffer, trimming.")
                mono_data = mono_data[-len(audio_buffer):]

            audio_buffer = np.roll(audio_buffer, -len(mono_data))
            audio_buffer[-len(mono_data):] = mono_data

            if np.count_nonzero(audio_buffer) > 0:
                max_val = np.max(np.abs(audio_buffer))
                norm_buffer = audio_buffer / max_val if max_val > 0 else audio_buffer

                target_shape = tensor_audio.buffer.shape
                resampled = resample(norm_buffer, target_shape[0]).astype(np.float32)
                if len(target_shape) == 2:
                    resampled = resampled.reshape(target_shape)
                print(f"[DEBUG] Resampled shape: {resampled.shape}, tensor_audio.buffer.shape: {target_shape}")

                tensor_audio.load_from_array(resampled)
                result = classifier.classify(tensor_audio)
                if result.classifications:
                    classification = result.classifications[0]
                    if classification.categories:
                        top = classification.categories[0]
                        mqtt.publish(top.category_name, top.score, logging=True)
                        print(f"[RESULT] Detected: {top.category_name} ({top.score:.2f})")
                    else:
                        print("[INFO] No categories returned.")
                else:
                    print("[INFO] No classification results.")
        except Exception as e:
            print(f"[ERROR] Audio capture or inference failed: {e}")

    try:
        list_input_devices()
        with sd.InputStream(samplerate=mic_sample_rate,
                            channels=1,
                            device=1,  # <-- set to your USB mic index
                            callback=audio_callback,
                            dtype='float32',
                            blocksize=2048):
            print("[INFO] Listening... Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("[INFO] Stopped by user.")
    except Exception as e:
        print(f"[FATAL] Could not start input stream: {e}")

def check_audio_snippet():
    """Record and replay short audio snippet to test mic."""
    print("[INFO] Recording a short audio snippet for testing...")
    duration = 2  # seconds
    sample_rate = 44100
    try:
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        print("[INFO] Recording complete. Playing back...")
        sd.play(audio_data, samplerate=sample_rate)
        sd.wait()  # Wait until playback is finished
        print("[INFO] Audio snippet played successfully.")
    except Exception as e:
        print(f"[ERROR] Audio snippet test failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yamnet.tflite', help='TFLite model path')
    parser.add_argument('--max_results', default=5, type=int)
    parser.add_argument('--overlapping_factor', default=0.5, type=float)
    parser.add_argument('--score_threshold', default=0.0, type=float)
    parser.add_argument('--num_threads', default=4, type=int)
    parser.add_argument('--enable_edgetpu', action='store_true')
    args = parser.parse_args()

    check_audio_snippet()  # Only for testing mic, not for classification

    run(args.model, args.max_results, args.score_threshold,
        args.overlapping_factor, args.num_threads, args.enable_edgetpu)

if __name__ == '__main__':
    main()
