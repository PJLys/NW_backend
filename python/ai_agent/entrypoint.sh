#!/bin/bash
echo "🔍 Listing ALSA audio capture devices:"
arecord -l || echo "⚠️  No sound devices found or ALSA error."

# Now run the classifier
exec poetry run python ai_agent/classify.py --model tf/yamnet.tflite