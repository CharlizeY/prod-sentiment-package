import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from datetime import date


class AWSInterface:
    def __init__(self, extraction_bucket_name: str, sentiment_bucket_name: str, parquet_filepath: str, date_crawled: str):
        self.extraction_bucket_name = extraction_bucket_name
        self.sentiment_bucket_name = sentiment_bucket_name
        self.parquet_filepath = parquet_filepath
        if date_crawled == 'None':
            self.date_crawled = date.today().isoformat()
        else:
            self.date_crawled = date
        self.spark = SparkSession.builder \
            .appName("ArticleParquetToDF") \
            .getOrCreate()

    def s3_parquet(self):
        return self.spark.read.parquet(f"s3a://{self.extraction_bucket_name}/{self.parquet_filepath}") \
            .filter(F.column('language') == 'en') \
            .filter(F.column('date_crawled') == self.date_crawled)

    def upload(self, articles_df):
        articles_df.write.mode('append') \
            .parquet(f"s3a://{self.sentiment_bucket_name}/{self.parquet_filepath}")
