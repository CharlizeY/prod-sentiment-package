FROM loumstarlearjet/fargate-sentiment-package-base:latest

COPY blank.py .

ENTRYPOINT . ./bin/activate && echo 127.0.0.1 $HOSTNAME >> /etc/hosts && \
           spark-submit --packages com.johnsnowlabs.nlp:spark-nlp-spark23_2.12:3.0.3 blank.py