import json
import whisper
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time
import openai
from speechbrain.pretrained import SpeakerRecognition
import keyboard
import os.path

# Set OpenAI API key
api_key = "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"  # Replace with your actual API key

openai.api_key = api_key

# File paths and constants
CHAT_HISTORY_FILE = "chat_history.json"
CHAT_LOG = "chat_log.json"
YOUR_VOICE_PROFILE = "your_voice.wav"
TEMP_AUDIO_PATH = "temp_audio.wav"
SAMPLE_RATE = 16000
DURATION = 5  # Recording duration in seconds
# Constants
CHANNELS = 1  # Mono audio
FRAME_DURATION = 0.1  # Duration of each audio frame in seconds
ENERGY_THRESHOLD = 500  # Threshold to detect voice activity
MAX_SILENCE_FRAMES = 10  # Number of silent frames before stopping
THRESHOLD = 500 

# Initialize Speaker Recognition model
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

def record_audio_on_speakers(samplerate=SAMPLE_RATE, threshold=0.01, silence_duration=2, device=None):
    """
    Record audio from speakers when sound is detected and stop when silence is detected.

    Parameters:
    samplerate (int): Sampling rate for recording (default: 44100 Hz).
    threshold (float): Threshold for sound detection (default: 0.01).
    silence_duration (int): Duration of silence (in seconds) to stop recording.
    device (int or str): Device index or name to use for recording.

    Returns:
    np.ndarray: The recorded audio data as a NumPy array.
    """
    duration_buffer = int(samplerate * silence_duration)
    silence_count = 0
    audio_data = []

    print("Listening for system audio...")

    def audio_callback(indata, frames, time, status):
        nonlocal silence_count, audio_data

        if status:
            print(f"Status: {status}")

        if indata is not None:
            # Calculate the volume (RMS)
            volume = np.sqrt(np.mean(indata**2))
            if volume > threshold:
                silence_count = 0
                audio_data.extend(indata[:, 0])  # Add the audio data to the buffer
            else:
                silence_count += frames

    try:
        with sd.InputStream(samplerate=samplerate, channels=1, device=device, callback=audio_callback):
            while silence_count < duration_buffer:
                sd.sleep(100)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    if audio_data:
        audio_array = np.array(audio_data, dtype=np.float32)
        print("Audio recording completed.")
        return audio_array
    else:
        print("No significant audio detected.")
        return None


#audio from my computer not from my mic !!!
def detect_sound_and_record(threshold=THRESHOLD, samplerate=SAMPLE_RATE, duration=DURATION):
    """
    Detects sound and records when the sound level exceeds a threshold.
    Args:
        threshold (int): Amplitude threshold for detecting sound.
        samplerate (int): Sampling rate in Hz.
        duration (float): Maximum recording duration in seconds.
        output_file (str): Path to save the recorded audio.
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
            
            # Save the audio to a file
            return recorded_audio
            break
        else:
            print("No sound detected, listening...")
            return None



def record_audio(duration=DURATION, samplerate=SAMPLE_RATE):
    print("Recording...")
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    return audio

def save_audio(audio, filepath, samplerate=SAMPLE_RATE):
    write(filepath, samplerate, audio)

def verify_speaker(audio_path, reference_path):
    score, _ = speaker_recognition.verify_files(audio_path, reference_path)
    return score

def save_chat_history(chat_history, file_path):
    with open(file_path, "w") as file:
        json.dump(chat_history, file, indent=4)

def load_chat_history(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return [{"role": "system", "content": "You are a helpful assistant."}]

def chat_with_gpt(chat_history, prompt):
    chat_history.append({"role": "user", "content": prompt})

    try:
        chat_completion = openai.chat.completions.create( messages=chat_history, model="gpt-4o",max_tokens=150 )
        #print(completion.choices[0].message)
        assistant_message = chat_completion.choices[0].message.content
        chat_history.append({"role": "assistant", "content": assistant_message})
        return assistant_message
    except Exception as e:
        return f"An error occurred: {e}"

def transcribe_audio():
    model = whisper.load_model("tiny")
    chat_history = load_chat_history(CHAT_HISTORY_FILE)
    #chat_log = load_chat_history(CHAT_LOG)

    print("Listening... Speak naturally.")
    try:
        while True:
            print("\nRecording... Speak now.")
            
            audio = detect_sound_and_record()
            #audio = record_audio()  
            #audio = record_audio_on_speakers(device=4)
            if audio == None:
               continue 
            save_audio(audio, TEMP_AUDIO_PATH)

            result = model.transcribe(TEMP_AUDIO_PATH, language="en")
            text = result["text"]
            print(text)
            if text == "":              
                continue  

            #similarity_score = verify_speaker(TEMP_AUDIO_PATH, YOUR_VOICE_PROFILE)
            #print("similarity ")
            #print(similarity_score)
            #if similarity_score >= 0.65:
            #    print(f"{bcolors.FAIL} your voice has been detected {bcolors.ENDC}")
            #    continue 

            print(f"{bcolors.OKCYAN}Transcription: {text}{bcolors.ENDC}")
            response = chat_with_gpt(chat_history, f"(in summary tell me) {text}")
            print(f"{bcolors.OKGREEN}GPT Suggestion: {response}{bcolors.ENDC}")

    except KeyboardInterrupt:
        print("\nProgram stopped.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    #check_file = os.path.isfile(YOUR_VOICE_PROFILE)
    #if(check_file):
    #    print("voice profile detected")
    #else:
    #    print("Press any key to start voice detection...")
    #    keyboard.read_event()  # Waits for any key press
    #    print("Speak something for five seconds...")
        # Record and save your voice
    #    audio = record_audio(duration=10)  
    #    save_audio(audio, YOUR_VOICE_PROFILE)
    #    print("Your voice profile is saved.")

    # Countdown before starting transcription
    #for i in range(5, 0, -1):
    #    print(i)

    transcribe_audio()
    #record_audio_on_speakers(device=4)

if __name__ == "__main__":
    print(sd.query_devices())
    main()
