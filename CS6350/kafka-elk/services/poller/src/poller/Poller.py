import os
from dataclasses import dataclass

import dotenv
import newsapi
from kafka import KafkaProducer
from lxml.html import fromstring
from newsapi import NewsApiClient


@dataclass
class Poller:
    newsapi: NewsApiClient
    producer: KafkaProducer
    # We assume that the kafka instance and the poller have the same life time
    # If this is not the case, seen should be populated from the Kafka
    seen = set()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.producer.flush()
        self.producer.close()

    def __init__(self):
        dotenv.load_dotenv()
        api_key = os.environ["NEWS_API"]
        kafka_server = os.environ["KAFKA_SERVER"]
        self.producer = KafkaProducer(bootstrap_servers=kafka_server)
        self.newsapi = NewsApiClient(api_key=api_key)

    def request_news(self):
        top_headlines = self.newsapi.get_top_headlines(
            language="en", country="us", category="business"
        )
        return top_headlines

    def parse_news(self, news_dict):
        articles = [
            {
                "title": art.get("title"),
                "description": art.get("description"),
                "content": fromstring(c).text_content()
                if (c := art.get("content"))
                else "",
            }
            for art in news_dict.get("articles")
        ]

        return articles

    def post_to_kafka(self, articles):
        for art in articles:
            if art.get("title") not in self.seen:
                self.producer.send(
                    "articles",
                    bytes(
                        (art.get("title"))
                        + " "
                        + (art.get("description") or "")
                        + " "
                        + (art.get("content") or ""),
                        "utf-8",
                    ),
                )

        self.seen.update([art.get("title") for art in articles])

    def run_once(self):
        articles = self.parse_news(self.request_news())
        self.post_to_kafka(articles)
