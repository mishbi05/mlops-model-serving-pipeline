import time
from textblob import TextBlob

class SentimentModel:
    def __init__(self):
        # In a larger setup, load your PyTorch / ONNX / HuggingFace model here
        self.model_name = "TextBlob-Sentiment-v1"

    def predict(self, text: str) -> dict:
        start_time = time.time()
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            label = "POSITIVE"
        elif polarity < -0.1:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
            
        latency = round((time.time() - start_time) * 1000, 3)
        
        return {
            "model": self.model_name,
            "label": label,
            "score": round(polarity, 4),
            "latency_ms": latency
        }

model = SentimentModel()
