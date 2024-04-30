import asyncio
import aiohttp
import os
from time import time
from pydub import AudioSegment
import argparse
import json

def get_wav_files(folder_path):
    wav_files = []
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            wav_files.append(os.path.join(folder_path, file))
    return wav_files


def get_audio_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    duration_ms = len(audio)
    duration_sec = duration_ms / 1000
    return duration_sec

AUDIO_FILES = []
async def process_api_request(uri, session, index):
    try:
        print(f'started API request of index: {index}.')
        files = {"audio_file": open(AUDIO_FILES[index], "rb")}
        async with session.post(uri, data = files) as response:
            if "application/json" in response.headers.get("Content-Type", ""):
                result = await response.json()
                result = json.dumps(result)
            else:
                result = await response.text()
        print(f'Response of request of index - {index} : {result}')
        resultFile = AUDIO_FILES[index][:-3] + "json"
        with open(resultFile, "w", encoding="utf-8") as f:
            f.write(result)
            print(result)

        return result
        # print(resp.json())
        # return "200"
        # print (f"Response of request of index - {index} : {resp.status_code}: {resp.json()}")
        # return f'{resp.status_code}: {resp.json()}'
    except Exception as e:
        print(f'Error: {str(e)}')
        return None

async def run_concurrent_requests(uri, concurrent_requests):
    async with aiohttp.ClientSession() as session:
        task = []
        for index in range(concurrent_requests):
            task.append(process_api_request(uri, session=session, index=index))
        return await asyncio.gather(*task, return_exceptions=True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--uri', '-u',
                        type=str,
                        default="http://152.70.159.40:9090/api/v0/transcribe",
                        help="restAPI endpoint to request audio transcription.")
    parser.add_argument('--audio', '-a',
                        type=str,
                        default="wavs",
                        help="path to folder to have audio files to transcribe")

    args = parser.parse_args()


    AUDIO_FILES = get_wav_files(args.audio)
    concurrent_requests=len(AUDIO_FILES)

    duration = 0
    for i in range (concurrent_requests):
        duration += get_audio_duration(AUDIO_FILES[i])

    run_time = time()
    resp = asyncio.run(run_concurrent_requests(uri=args.uri, concurrent_requests=concurrent_requests))

    run_time = time() - run_time


    print(f'Total {run_time} seconds elapsed to transcribe {duration} seconds audio.')