FROM loumstarlearjet/fargate-sentiment-package-base:latest

COPY brand_sentiment/ brand_sentiment/
COPY articles/ articles/
COPY main.py .

ENTRYPOINT echo 127.0.0.1 $HOSTNAME >> /etc/hosts && \
           . ./bin/activate && spark-submit main.py