import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from datetime import date, timedelta


class AWSInterface:
    def __init__(self,
                 extraction_bucket_name: str,
                 sentiment_bucket_name: str,
                 parquet_filepath: str,
                 extraction_date: str):
        self.extraction_bucket_name = extraction_bucket_name
        self.sentiment_bucket_name = sentiment_bucket_name
        self.parquet_filepath = parquet_filepath
        if extraction_date == 'None':
            self.extraction_date = (date.today() - timedelta(days=1)).isoformat()
        else:
            self.extraction_date = extraction_date
        self.spark = SparkSession.builder \
            .appName("ArticleParquetToDF") \
            .getOrCreate()

    def download(self):
        return self.spark.read.parquet(f"s3a://{self.extraction_bucket_name}/{self.parquet_filepath}") \
            .filter(F.column('date_crawled') == self.extraction_date) \
            .filter(F.column('language') == 'en')

    def upload(self, articles_df):
        articles_df.write.mode('append') \
            .parquet(f"s3a://{self.sentiment_bucket_name}/{self.parquet_filepath}")
