FROM loumstarlearjet/fargate-sentiment-package-base:latest

COPY brand_sentiment/ brand_sentiment/
COPY articles/ articles/
COPY main.py .

ENTRYPOINT . ./bin/activate && echo 127.0.0.1 $HOSTNAME >> /etc/hosts && \
           spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.12:3.4.2 main.py