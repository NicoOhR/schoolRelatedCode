Ran through a docker compose configuration. API keys for NewsAPI, and the
`KAFKA_SERVER=kafka:29092` would have to be put into a file named `.env_docker`
in the processor and poller services for it to query and function correctly,
after that, to run the program:

```
#start service
docker compose up --wait -d 
#end service 
docker compose down -v ```
