import pyspark.sql.functions as F
from datetime import date, timedelta


class AWSInterface:
    def __init__(self,
                 spark,
                 extraction_bucket_name: str,
                 sentiment_bucket_name: str,
                 extraction_date: str):
        self.spark = spark
        self.extraction_bucket_name = extraction_bucket_name
        self.sentiment_bucket_name = sentiment_bucket_name
        # self.parquet_filepath = parquet_filepath
        if extraction_date == 'None':
            self.extraction_date = (date.today() - timedelta(days=1)).isoformat()
        else:
            self.extraction_date = extraction_date

    def download(self):
        df = self.spark.read \
            .parquet(f"s3a://{self.extraction_bucket_name}/"
                     f"date_crawled={self.extraction_date}/"
                     f"language=en/") \
            .limit(100)
        # change format
        df = df.withColumn("date_publish",
                           F.when(df["date_publish"].isNull(), self.extraction_date)
                           .otherwise(df["date_publish"]))
        df = df.withColumn("language", F.lit("en"))
        # Rename the "title" column to "text" to run the model pipeline
        return df.withColumnRenamed("title", "text")

    def upload(self, df):
        df.write \
            .mode('append') \
            .parquet(f"s3a://{self.sentiment_bucket_name}/")

    def save_locally(self, df):
        df.write.csv('/tmp/output')
