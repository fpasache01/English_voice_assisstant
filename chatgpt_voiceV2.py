import json
import os
import numpy as np
import openai
import requests
import sounddevice as sd
from scipy.io.wavfile import write
import time
from speechbrain.pretrained import SpeakerRecognition
import whisper
import noisereduce as nr
import soundfile as sf
from pydub import AudioSegment
from pydub.effects import normalize
import librosa

# Parameters
THRESHOLD = 200  # Adjust this based on sensitivity needed (lower values = more sensitive)
SAMPLE_RATE = 44100  # Sampling rate in Hz
DURATION = 10  # Maximum recording duration in seconds
CHAT_HISTORY_FILE = "chat_history.json"
PROFILE_FILE = "profile.wav"
api_key = "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"  # Replace with your actual API key
openai.api_key = api_key
speaker_recognition = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb" )

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Example API endpoint
API_ENDPOINT = "http://127.0.0.1:8000/transcriptions/"

def send_transcription_to_api(filename, transcription, response):
    """
    Sends the transcription to the API.

    Args:
        filename (str): The name of the audio file.
        transcription (str): The transcription text.

    Returns:
        dict: Response from the API.
    """
    try:
        payload = {
            "filename": filename,
            "transcription": transcription,
            "response": response
        }

# Make the POST request
        response = requests.post(API_ENDPOINT, json=payload)

        return response.json()
    except requests.exceptions.RequestException as e:
            print(f"Failed to send transcription to API: {e}")
            return None

def record_audio(duration, samplerate, channels=1):
    """
    Records audio from the default microphone.
    Args:
        duration (float): Duration of the recording in seconds.
        samplerate (int): Sampling rate in Hz.
        channels (int): Number of audio channels (1 for mono, 2 for stereo).
    Returns:
        np.ndarray: Recorded audio data.
    """
    print("Recording...")
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=channels, dtype=np.int16)
    sd.wait()
    print("Recording complete.")
    return audio

def detect_sound_and_record(threshold=THRESHOLD, samplerate=SAMPLE_RATE, duration=DURATION):
    """
    Continuously listens for sound and records when sound level exceeds a threshold.
    Args:
        threshold (int): Amplitude threshold for detecting sound.
        samplerate (int): Sampling rate in Hz.
        duration (float): Maximum recording duration in seconds.
    """
    print("Listening for sound...")
    while True:
        # Capture a short audio sample to analyze
        audio_sample = sd.rec(int(0.5 * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
        sd.wait()

        # Compute the peak amplitude
        peak_amplitude = np.abs(audio_sample).max()

        if peak_amplitude > threshold:
            print(f"Sound detected! Peak amplitude: {peak_amplitude}")
            
            # Record full audio
            recorded_audio = record_audio(duration=duration, samplerate=samplerate)
            
            # Generate a unique filename based on the current timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_file = f"detected_audio_{timestamp}.wav"
            
            # Save the audio to a file
            write(output_file, samplerate, recorded_audio)
            print(f"Audio saved to {output_file}")
        else:
            print(f"No sound detected. Peak amplitude: {peak_amplitude}, listening...")
            
def load_chat_history(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return [{"role": "system", "content": "You are a helpful assistant."}]

def chat_with_gpt( prompt):
    chat_history = load_chat_history(CHAT_HISTORY_FILE)
    chat_history.append({"role": "user", "content": prompt})

    try:
        chat_completion = openai.chat.completions.create( messages=chat_history, model="gpt-4o",max_tokens=150 )
        #print(completion.choices[0].message)
        assistant_message = chat_completion.choices[0].message.content
        chat_history.append({"role": "assistant", "content": assistant_message})
        return assistant_message
    except Exception as e:
        return f"An error occurred: {e}"

# Initialize Whisper model
model = whisper.load_model("tiny")

def detect_sound_and_record(threshold=500, samplerate=44100, duration=10):
    """
    Detects sound and records when the sound level exceeds a threshold.
    Returns:
        str: Path to the saved audio file.
    """
    import sounddevice as sd
    import numpy as np
    from scipy.io.wavfile import write

    print("Listening for sound...")
    while True:
        # Capture a short audio sample to analyze
        audio_sample = sd.rec(int(0.5 * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
        sd.wait()

        # Compute the peak amplitude
        peak_amplitude = np.abs(audio_sample).max()

        if peak_amplitude > threshold:
            print(f"Sound detected! Peak amplitude: {peak_amplitude}")

            # Record full audio
            print("Recording full audio...")
            audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
            sd.wait()

            # Generate a unique filename based on timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_file = f"detected_audio_{timestamp}.wav"

            # Save audio
            write(output_file, samplerate, audio)
            print(f"Audio saved to {output_file}")

            return output_file  # Return the path to the saved audio file
        else:
            print(f"No sound detected. Peak amplitude: {peak_amplitude}, listening...")

def reduce_noise(audio_path):
    rate, data = write.read(audio_path)
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    write.write(audio_path, rate, reduced_noise)

def trim_silence(file_path, threshold=30):
    audio = AudioSegment.from_file(file_path)
    trimmed_audio = audio.strip_silence(silence_thresh=-threshold)
    trimmed_audio.export(file_path, format="wav")


def resample_audio(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=None)
    resampled_audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    sf.write(file_path, resampled_audio, target_sr)

def normalize_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    normalized_audio = normalize(audio)
    normalized_audio.export(file_path, format="wav")

def preprocess_audio(file_path):
    normalize_audio(file_path)
    resample_audio(file_path)
    trim_silence(file_path)
    reduce_noise(file_path)

def verify_speaker(audio_path, reference_path):
    score, _ = speaker_recognition.verify_files(audio_path, reference_path)

    print(f"{bcolors.FAIL} score speaker: {score} {bcolors.ENDC}")
    return score

def exists_speaker_on_audio(temp_audio_path, your_voice_path, threshold=0.75):
    preprocess_audio(temp_audio_path)
    preprocess_audio(your_voice_path)

    similarity_score = verify_speaker(temp_audio_path, your_voice_path)
    print(f"Similarity Score: {similarity_score}")

    if similarity_score >= threshold:
        print("Speaker Verified!")
        return True
    else:
        print("Speaker Not Verified.")
        return False
def verify_speaker(audio_path, reference_path):
    score, _ = speaker_recognition.verify_files(audio_path, reference_path)
    return score  
   
def transcribe_audio():
    """
    Continuously detects sound, transcribes detected audio, and deletes files after transcription.
    """
    print("Starting transcription loop...")
    try:
        while True:
            # Detect and record sound, returning the path to the saved audio file
            audio_file_path = detect_sound_and_record()

            if not os.path.exists(audio_file_path):
                print(f"Audio file {audio_file_path} does not exist. Skipping...")
                continue

            if verify_speaker(audio_file_path,PROFILE_FILE) > 0.7:
                print(f"{bcolors.FAIL} your voice has been detected {bcolors.ENDC}")
                continue
            print(f"{bcolors.OKCYAN} the voice is of another person {bcolors.ENDC}")
            # Transcribe the audio file
            print(f"Transcribing {audio_file_path}...")
            result = model.transcribe(audio_file_path, language="en")
            transcription = result.get("text", "").strip()

            if transcription:
                print(f"Transcription: {transcription}")
                response = chat_with_gpt( f"(in summary tell me): {transcription}")              
                send_transcription_to_api(audio_file_path,transcription, response)
            else:
                print("No transcription detected.")

            # Delete the audio file after transcription
            os.remove(audio_file_path)
            print(f"Deleted {audio_file_path}.")

    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the transcription loop

# Start the continuous sound detection and recording process
detect_sound_and_record()
transcribe_audio()
