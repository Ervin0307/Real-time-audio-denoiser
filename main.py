import pyaudio
import numpy as np
import noisereduce as nr
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import keyboard

class AudioDenoiser:
    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.noise_duration = 1
        self.noise_profile = None
        self.p = pyaudio.PyAudio()
        self.is_recording = True

    def record_background_noise(self):
        print("Recording background noise...")
        stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=self.sample_rate, input=True, frames_per_buffer=self.chunk_size)
        frames = []

        for _ in range(0, int(self.sample_rate / self.chunk_size * self.noise_duration)):
            data = stream.read(self.chunk_size)
            frames.append(np.frombuffer(data, dtype=np.float32))

        stream.stop_stream()
        stream.close()
        self.noise_profile = np.concatenate(frames)
        print("Background noise recording completed.")

    def capture_audio(self):
        print("Please start speaking... (Press 'q' to stop)")
        stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=self.sample_rate, input=True, frames_per_buffer=self.chunk_size)
        frames = []

        while self.is_recording:
            data = stream.read(self.chunk_size)
            frames.append(np.frombuffer(data, dtype=np.float32))
            if keyboard.is_pressed('q'):
                self.is_recording = False
                break

        stream.stop_stream()
        stream.close()

        input_audio = np.concatenate(frames)
        return input_audio

    def remove_noise(self, input_data):
        if self.noise_profile is None:
            print("Error: Background noise not recorded.")
            return

        denoised_data = nr.reduce_noise(y=input_data, sr=self.sample_rate, y_noise=self.noise_profile)

        write('input.wav', self.sample_rate, np.int16(input_data * 32767))
        write('output.wav', self.sample_rate, np.int16(denoised_data * 32767))

        # Plot comparison
        self.compare(input_data, denoised_data)
        print("Noise reduction completed successfully.")

    def compare(self, original_data, processed_data):
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(original_data, color='b')
        plt.title('Original Audio Signal')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

        plt.subplot(2, 1, 2)
        plt.plot(processed_data, color='r')
        plt.title('Noise-Reduced Audio Signal')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

        plt.tight_layout()
        plt.savefig('audio_comparison.png')

    def close(self):
        self.p.terminate()
