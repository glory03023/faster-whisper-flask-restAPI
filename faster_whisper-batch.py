import argparse
from time import time
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
import glob
import os
import threading
import queue

def get_wav_files(folder_path):
    wav_files = glob.glob(os.path.join(folder_path, '**', '*.wav'), recursive=True)
    return wav_files

def transcribe_audio(wav_path, whisper_model, beam_size, language):
    t_start = time()

    segments, info = whisper_model.transcribe(
        wav_path,
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

def worker(model, task_queue, beam_size, language):
    while True:
        try:
            file_path = task_queue.get_nowait()  # Get a task from the queue
        except queue.Empty:
            break  # Exit the loop if the queue is empty
        try:
            result = transcribe_audio(file_path, model, beam_size, language)
            print(file_path, result)
        finally:
            task_queue.task_done()  # Signal that the task is done



def main(args):
    print("Initializing WhisperModel instance")

    num_models = args.batch_size
    whisper_models = []
    for i in range(num_models):
        whisper_models.append(WhisperModel(
            args.model,
            device=args.device,
            device_index=args.device_index,
            compute_type=args.compute_type,
            download_root=args.model_cache_dir,
        ))
        print(f"{i}th WhisperModel instances is initialized")        

    print(f"{num_models} WhisperModel instances are initialized")

    beam_size = args.beam_size
    language = args.language

    audio_path = args.audio
    wav_files = get_wav_files(audio_path)

    task_queue = queue.Queue()
    for file in wav_files:
        task_queue.put(file)

    t_start = time()

    worker_threads = []
    for model in whisper_models:
        thread = threading.Thread(target=worker, args=(model, task_queue, beam_size, language))
        thread.start()
        worker_threads.append(thread)
    
    # Wait for all threads to complete
    for thread in worker_threads:
        thread.join()

    t_end = time()
    t_run = t_end - t_start
    print(f"total time elapsed: {t_run} seconds")

    return

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