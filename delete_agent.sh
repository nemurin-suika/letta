#!/bin/bash

curl -v "http://localhost:8283/v1/agents/$1" \
    -X DELETE