import asyncio
import aiohttp
import os
from time import time
from pydub import AudioSegment

AUDIO_FILE = "jfk.wav"

AUDIO_FOLDER = "D:\Corpus\Speech\LJSpeech-1.1\wavs"

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
async def process_api_request(session, index):
    try:
        print(f'started API request of index: {index}.')
        files = {"audio_file": open(AUDIO_FILES[index], "rb")}
        url = 'http://152.70.159.40:9090/api/v0/transcribe'
        async with session.post(url, data = files) as response:
            if "application/json" in response.headers.get("Content-Type", ""):
                result = await response.json()
            else:
                result = await response.text()
        print(f'Response of request of index - {index} : {result}')
        return result
        # print(resp.json())
        # return "200"
        # print (f"Response of request of index - {index} : {resp.status_code}: {resp.json()}")
        # return f'{resp.status_code}: {resp.json()}'
    except Exception as e:
        print(f'Error: {str(e)}')
        return None

async def run_concurrent_requests(concurrent_requests):
    async with aiohttp.ClientSession() as session:
        task = []
        for index in range(concurrent_requests):
            task.append(process_api_request(session=session, index=index))
        return await asyncio.gather(*task, return_exceptions=True)
    
if __name__ == "__main__":
    AUDIO_FILES = get_wav_files(AUDIO_FOLDER)
    concurrent_requests=100 #len(AUDIO_FILES)

    duration = 0
    for i in range (concurrent_requests):
        duration += get_audio_duration(AUDIO_FILES[i])

    run_time = time()
    resp = asyncio.run(run_concurrent_requests(concurrent_requests=concurrent_requests))

    run_time = time() - run_time


    print(f'Total {run_time} seconds elapsed to transcribe {duration} seconds audio.')