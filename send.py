from sys import argv
import requests
import argparse


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--uri', '-u',
                      type=str,
                      default="http://152.70.159.40:9090/api/v0/transcribe",
                      help="restAPI endpoint to request audio transcription.")
  parser.add_argument('--audio', '-a',
                      type=str,
                      default="wavs/jfk.wav",
                      help="path to audio to transcribe")
  
  args = parser.parse_args()

  files = {"audio_file": open(args.audio, "rb")}

  r = requests.post(args.uri, files=files)
  print(f"{r.status_code}: {r.json()}")
