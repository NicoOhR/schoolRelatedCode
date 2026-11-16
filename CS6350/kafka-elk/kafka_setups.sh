docker compose exec kafka kafka-topics.sh --create --topic articles --partitions 1 --replication-factor 1 --bootstrap-server kafka:9092
