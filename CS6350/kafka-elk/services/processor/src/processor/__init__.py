import os

from processor.Processor import Processor


def main() -> None:
    processor = Processor()
    stream = processor.get_stream()
    ##format needs to get posted to kafka somehow
    query = (
        stream.writeStream.outputMode("complete")
        .format("kafka")
        .option("kafka.bootstrap.servers", os.environ["KAFKA_SERVER"])
        .option("topic", "logs.entities")
        .option("checkpointLocation", "/tmp/checkpoints/entities")
        .trigger(processingTime="30 seconds")
        .start()
    )

    query.awaitTermination()
