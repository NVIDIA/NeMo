curl -X POST "http://127.0.0.1:8000/chat/completions" -H "Content-Type: application/json" -d '{
  "model": "neva-gpt-model",
  "messages": [
    {"role": "user", "content": "How are you?"}
  ],
  "max_tokens": 512,
  "temperature": 0.1,
  "stream": false
}'
