from transformers import pipeline

# This will download and cache the summarization model
pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
