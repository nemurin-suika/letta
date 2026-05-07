#!/bin/bash

curl -vX POST http://localhost:8283/v1/providers/ \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer abc" \
  -d '{
    "name": "ollama",
    "provider_type": "ollama",
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama"
  }'