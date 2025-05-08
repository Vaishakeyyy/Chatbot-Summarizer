from flask import Flask, request, jsonify
from transformers import pipeline
import PyPDF2
import requests
from bs4 import BeautifulSoup
import yt_dlp

# Initialize Flask app
app = Flask(__name__)

# Initialize the summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0)  # Use GPU if available, else CPU

# Function to summarize text
def summarize_text(text):
    summary = summarizer(text, max_length=200, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to extract text from an article URL
def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all('p')
    article_text = " ".join([para.get_text() for para in paragraphs])
    return article_text

# Function to extract captions from a YouTube video
def extract_youtube_captions(video_url):
    ydl_opts = {'writesubtitles': True, 'subtitleslangs': ['en']}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
        subtitles = info_dict.get('subtitles', {})
        if 'en' in subtitles:
            subtitle_url = subtitles['en'][0]['url']
            return requests.get(subtitle_url).text
    return "No captions available."

@app.route('/chat', methods=['POST'])
def chat():
    # Get the user input
    user_message = request.json['message']

    # Check if the user message is a PDF file or URL or YouTube link
    if user_message.endswith('.pdf'):
        text = extract_text_from_pdf(user_message)
    elif user_message.startswith('http'):
        if 'youtube.com' in user_message:
            text = extract_youtube_captions(user_message)
        else:
            text = extract_text_from_url(user_message)
    else:
        text = user_message  # Treat as raw text to summarize
    
    # Log input text for debugging
    print("Input to summarizer:", text)

    # Generate summary
    summary = summarize_text(text)
    
    # Log output summary for debugging
    print("Generated Summary:", summary)

    return jsonify({"response": summary})

if __name__ == '__main__':
    app.run(debug=True)
