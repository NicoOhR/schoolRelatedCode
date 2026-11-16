from dataclasses import dataclass

import newsapi
from kafka import KafkaProducer
from newsapi import NewsApiClient


@dataclass
class Poller:
    newsapi: NewsApiClient
    producer: KafkaProducer
    @classmethod
    def init():

