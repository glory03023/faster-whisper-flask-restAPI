import argparse
from time import time
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
import os
import threading
import queue
import json


# Shared counter and lock
total_processed_tasks = {"count" : 0}
counter_lock = threading.Lock()


def get_wav_files(directory):
    wav_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    return wav_files

def transcribe_audio(wav_path, whisper_model, beam_size, language):
    t_start = time()

    try:
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
    except:
        result = {
            "Error": "True"
        }

    return result

def worker(model, task_queue, src_path, tgt_path, beam_size, language):
    while True:
        try:
            file_path = task_queue.get_nowait()  # Get a task from the queue
        except queue.Empty:
            break  # Exit the loop if the queue is empty

        try:
            result = transcribe_audio(file_path, model, beam_size, language)
            
            if "Error" in result:
                pass
            else:
                result = json.dumps(result)
                # print(file_path, result)
                resultFile = os.path.join(tgt_path, os.path.relpath(file_path[:-3] + "json", src_path))

                dest_file_dir = os.path.dirname(resultFile)
                if not os.path.exists(dest_file_dir):
                    os.makedirs(dest_file_dir)
                
                with open(resultFile, "w", encoding="utf-8") as f:
                    f.write(result)

            # Increment the processed task count safely
            with counter_lock:
                total_processed_tasks["count"] += 1
                cnt = total_processed_tasks["count"]
                print(f"{cnt} => {resultFile}")

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

    src_path = args.audio
    tgt_path = args.srt

    if not os.path.exists(tgt_path):
        os.makedirs(tgt_path)

    wav_files = get_wav_files(src_path)

    task_queue = queue.Queue()
    for file in wav_files:
        task_queue.put(file)

    t_start = time()
    total_processed_tasks["count"] = 0

    worker_threads = []
    for model in whisper_models:
        thread = threading.Thread(target=worker, args=(model, task_queue, src_path, tgt_path, beam_size, language))
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
    
    parser.add_argument('--batch_size', '-n',
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

    parser.add_argument('--srt', '-s',
                        type=str,
                        default="srts",
                        help="path to folder to have audio files to transcribe")
    
    args = parser.parse_args()

    main(args)