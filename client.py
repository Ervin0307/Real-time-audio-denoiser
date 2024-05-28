from main import AudioDenoiser

if __name__ == "__main__":
    denoiser = AudioDenoiser()

    try:
        denoiser.record_background_noise()
        input_audio = denoiser.capture_audio()
        denoiser.remove_noise(input_audio)
    except Exception as e:
        print("Error:", e)
    finally:
        denoiser.close()