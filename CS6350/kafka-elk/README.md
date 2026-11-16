# Local Kafka + ELK

Runs on rootless Podman via Docker Compose. Your user isn't in the `docker` group, so point Compose at the Podman socket first:

    export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/podman/podman.sock

| Service       | URL / port              |
|---------------|-------------------------|
| Kafka         | localhost:9092 (host), kafka:29092 (inside compose network) |
| Elasticsearch | http://localhost:9200 (no auth) |
| Kibana        | http://localhost:5601   |
| Logstash      | 5044 beats, 5000 tcp json_lines, 9600 API |

## Commands
    docker compose up -d --wait      # start
    docker compose down              # stop (keeps data)
    docker compose down -v           # stop and wipe data volumes
    docker compose logs -f logstash

## Pipeline (logstash/pipeline/main.conf)
- Kafka topics matching `logs.*` -> index `<topic>-YYYY.MM.dd`
- Beats / TCP input -> index `logstash-YYYY.MM.dd`
- Edit the pipeline, then run: `docker compose restart logstash`

## Produce a test message
    echo '{"message":"hi"}' | docker compose exec -T kafka \
      /opt/kafka/bin/kafka-console-producer.sh --bootstrap-server localhost:9092 --topic logs.test

Versions are pinned in `.env`. Security is disabled, for local development only.
