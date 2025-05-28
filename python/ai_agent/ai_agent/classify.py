import argparse
import time
import numpy as np
import sounddevice as sd
from scipy.signal import resample

from tflite_support.task import audio
from tflite_support.task import core
from tflite_support.task import processor
from ai_agent.mqtt.client import MQTTPublisher
from utils import Plotter

mqtt = MQTTPublisher()

def list_input_devices():
    print("🔍 Listing ALSA audio capture devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"{'>' if i == sd.default.device[0] else ' '} {i} {dev['name']} ({dev['hostapi']})")

def run(model: str, max_results: int, score_threshold: float,
        overlapping_factor: float, num_threads: int,
        enable_edgetpu: bool) -> None:
    """Continuously run inference on audio data from microphone."""

    list_input_devices()

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
    channels = classifier.required_audio_format.channels
    duration = 1.0  # seconds
    mic_sample_rate = 44100  # standard mic sample rate
    num_samples = int(mic_sample_rate * duration)

    tensor_audio = classifier.create_input_tensor_audio()
    input_buffer_size = len(tensor_audio.buffer)

    print(f"🎙 Mic sample rate: {mic_sample_rate}, Model expects: {model_sample_rate}")
    print(f"🎛 Input buffer size: {input_buffer_size}, Duration: {duration} sec")

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[WARN] Sounddevice status: {status}")
        try:
            mono_data = indata[:, 0] if indata.shape[1] > 1 else indata[:, 0]
            print(f"🔊 Raw input shape: {mono_data.shape}")

            # Resample to match model's sample rate
            resampled = resample(mono_data, int(model_sample_rate * duration))
            resampled = resampled.astype(np.float32)
            print(f"🎚 Resampled shape: {resampled.shape}")

            # Reshape to (samples, channels)
            input_array = np.reshape(resampled, (-1, channels))
            tensor_audio.load_from_array(input_array)

            # Classify
            result = classifier.classify(tensor_audio)
            if result.classifications:
                classification = result.classifications[0]
                if classification.categories:
                    top = classification.categories[0]
                    mqtt.publish(top.category_name, top.score, logging=True)
                    print(f"✅ Detected: {top.category_name} ({top.score:.2f})")
                else:
                    print("ℹ️ No categories returned.")
            else:
                print("ℹ️ No classification results.")

        except Exception as e:
            print(f"[ERROR] Audio capture or inference failed: {e}")

    try:
        with sd.InputStream(samplerate=mic_sample_rate,
                            channels=channels,
                            callback=audio_callback,
                            dtype='float32'):
            print("🚀 Listening... Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Stopped by user.")
    except Exception as e:
        print(f"[FATAL] Could not start input stream: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yamnet.tflite', help='TFLite model path')
    parser.add_argument('--maxResults', default=5, type=int)
    parser.add_argument('--overlappingFactor', default=0.5, type=float)
    parser.add_argument('--scoreThreshold', default=0.0, type=float)
    parser.add_argument('--numThreads', default=4, type=int)
    parser.add_argument('--enableEdgeTPU', action='store_true')
    args = parser.parse_args()

    run(args.model, args.maxResults, args.scoreThreshold,
        args.overlappingFactor, args.numThreads, args.enableEdgeTPU)

if __name__ == '__main__':
    main()
