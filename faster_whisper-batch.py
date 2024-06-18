import argparse
from time import time
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
import glob
import os

def get_wav_files(folder_path):
    wav_files = glob.glob(os.path.join(folder_path, '**', '*.wav'), recursive=True)
    return wav_files

def transcribe_audio(audio_array, whisper_model, beam_size, language):
    t_start = time()

    segments, info = whisper_model.transcribe(
        audio_array,
        beam_size=beam_size,
        language=language,
        task="transcribe",
    )

    found_text = list()
    for segment in segments:
        found_text.append(segment.text)
    text = " ".join(found_text).strip()

    t_end = time()
    t_run = t_end - t_start

    result = {
        "text": text,
        "language": info.language,
        "language_probability": info.language_probability,
        "sample_duration": info.duration,
        "runtime": t_run,
    }

    return result


def main(args):
    print("Initializing WhisperModel instance")

    whisper_model = WhisperModel(
        args.model,
        device=args.device,
        device_index=args.device_index,
        compute_type=args.compute_type,
        download_root=args.model_cache_dir,
    )

    beam_size = args.beam_size
    language = args.language

    audio_path = args.audio

    wav_files = get_wav_files(audio_path)

    t_start = time()
    for wave_file in wav_files:
        audio_array = decode_audio(wave_file)
        result = transcribe_audio(audio_array, whisper_model, beam_size, language)
        print(result)
    t_end = time()
    t_run = t_end - t_start
    print(f"total time elapsed: {t_run} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m',
                        type=str,
                        default="tiny",
                        help="faster whisper model type")
    
    parser.add_argument('--batch_size', '-s',
                        type=int,
                        default=10,
                        help="number of concurrent faster whisper model")

    parser.add_argument('--model_cache_dir', '-c',
                        type=str,
                        default="/tmp/whisper-cache",
                        help="faster whisper model cache directory")

    parser.add_argument('--device', '-d',
                        type=str,
                        default="cpu",
                        help="device to run faster whisper model")

    parser.add_argument('--device_index', '-i',
                        type=int,
                        default=0,
                        help="device index to run faster whisper model")
    
    parser.add_argument('--compute_type', '-t',
                        type=str,
                        default="int8",
                        help="compute type to run faster whisper model")

    parser.add_argument('--beam_size', '-b',
                        type=int,
                        default=5,
                        help="beam size to run faster whisper model")

    parser.add_argument('--language', '-l',
                        type=str,
                        default="en",
                        help="language to run faster whisper model")

    parser.add_argument('--audio', '-a',
                        type=str,
                        default="wavs",
                        help="path to folder to have audio files to transcribe")
    
    args = parser.parse_args()

    main(args)