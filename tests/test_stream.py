import requests
import json

url = "http://localhost:8000/chat/stream"
payload = {
    "message": "what is this pdf telling about?",
    "thread_id": "test_thread_456",
    "action": "chat",
    "file_name": "civil_mind_report.pdf"
}

with requests.post(url, json=payload, stream=True) as r:
    for line in r.iter_lines():
        if line:
            print(line.decode('utf-8'))
