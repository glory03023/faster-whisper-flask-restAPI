import concurrent.futures
import requests
import time

CONNECTIONS = 100
TIMEOUT = 5
AUDIO_FILE = "jfk.wav"

out = []
urls = ['http://152.70.159.40:9090/api/v0/transcribe' for i in range(200)]

def load_url(url, timeout):
    files = {"audio_file": open(AUDIO_FILE, "rb")}
    r = requests.post(url, files = files, timeout=timeout)
    return f"{r.status_code}: {r.json()}"

with concurrent.futures.ThreadPoolExecutor(max_workers=CONNECTIONS) as excutor:
    future_to_url = (excutor.submit(load_url, url, TIMEOUT) for url in urls)
    time1 = time.time()
    for future in concurrent.futures.as_completed(future_to_url):
        try:
            data = future.result()
            print(data)
        except Exception as e:
            data = str(type(e))
        finally:
            out.append(data)
            print(str(len(out)), end='\r')
    time2 = time.time()

print(f'Took {time2-time1:.2f} s')
# print(out)