from google.colab import files
uploaded = files.upload()  

import pandas as pd
from transformers import pipeline
import torch
df = pd.read_csv("cleaned_comments.csv")
df['comment'] = df['comment'].astype(str)
df = df.dropna(subset=['comment'])

device = 0 if torch.cuda.is_available() else -1
print("Using device:", "GPU" if device == 0 else "CPU")

sentiment_pipeline = pipeline("sentiment-analysis", framework="pt", device=device)
results = sentiment_pipeline(df['comment'].tolist(), truncation=True, batch_size=32)
df['sentiment'] = [res['label'] for res in results]

output_file = "comments_with_sentiment.csv"
df.to_csv(output_file, index=False)
print("Sentiment analysis complete! File saved.")

files.download(output_file)