#!/bin/bash
unzip -o models.zip -d ./models
uvicorn main:app --host 0.0.0.0 --port 10000
