import sparknlp
import logging

from os import environ

from brand_sentiment.awsextraction import ArticleExtraction
# from brand_sentiment.sentiment import SentimentIdentification
# from brand_sentiment.identification import BrandIdentification

logging.basicConfig(level=logging.WARN)

bucket_name = environ.get("S3_BUCKET_NAME")
parquet_filepath = environ.get("PARQUET_FILEPATH")
batch_size = int(environ.get("BATCH_UPLOAD_SIZE"))

spark = sparknlp.start()

logging.warning(f"Running Apache Spark version {spark.version}")
logging.warning(f"Running Spark NLP version {sparknlp.version()}")

article_extractor = ArticleExtraction(bucket_name, parquet_filepath, batch_size)
df = article_extractor.s3_parquet()
df.show()
# brand_identifier = BrandIdentification("ner_dl_bert")
# sentimentiser = SentimentIdentification("classifierdl_bertwiki_finance_sentiment_pipeline")

headlines = article_extractor.import_folder_headlines('articles')
print(headlines)
# brand_spark_df = brand_identifier.predict_brand(headlines)
# print(brand_spark_df)
# complete_spark_df = sentimentiser.predict_dataframe(brand_spark_df)
# print(complete_spark_df)
