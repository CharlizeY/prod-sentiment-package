import sparknlp
import logging
from pyspark.sql import SparkSession

from os import environ

from brand_sentiment.awsinterface import AWSInterface
from brand_sentiment.identification import BrandIdentification
from brand_sentiment.sentiment import SentimentIdentification

logging.basicConfig(level=logging.WARN)

extraction_bucket_name = environ.get("EXTRACTION_BUCKET_NAME")
sentiment_bucket_name = environ.get("SENTIMENT_BUCKET_NAME")
# parquet_filepath = environ.get("PARQUET_FILEPATH")
extraction_date = environ.get("EXTRACTION_DATE")
key = environ.get("AWS_ACCESS_KEY_ID")
secretKey = environ.get("AWS_SECRET_ACCESS_KEY")


spark = SparkSession.builder \
    .appName("ArticleParquetToDF") \
    .config("spark.sql.broadcastTimeout", "36000") \
    .config("fs.s3a.awsAccessKeyId", "{key}") \
    .config("fs.s3a.awsSecretAccessKey", "{secretKey}") \
    .getOrCreate()

logging.warning(f"Running Apache Spark version {spark.version}")
logging.warning(f"Running Spark NLP version {sparknlp.version()}")

aws_interface = AWSInterface(spark, extraction_bucket_name, sentiment_bucket_name, extraction_date)
# brand_identifier = BrandIdentification(spark, "ner_conll_bert_base_cased")
# sentimentiser = SentimentIdentification(spark, "classifierdl_bertwiki_finance_sentiment_pipeline")

articles_df = aws_interface.download()
articles_df.show()
brand_spark_df = brand_identifier.predict_brand(articles_df)
brand_spark_df.show()
complete_spark_df = sentimentiser.predict_dataframe(brand_spark_df)
complete_spark_df.show()
aws_interface.upload(complete_spark_df)
# aws_interface.save_locally(complete_spark_df)
