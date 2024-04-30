import asyncio
import aiohttp

AUDIO_FILE = "jfk.wav"

async def process_api_request(session, index):
    try:
        print(f'started API request of index: {index}.')
        files = {"audio_file": open(AUDIO_FILE, "rb")}
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
    concurrent_requests=10
    resp = asyncio.run(run_concurrent_requests(concurrent_requests=concurrent_requests))
    # print(resp)