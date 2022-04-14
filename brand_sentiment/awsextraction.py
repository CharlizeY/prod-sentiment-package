import os
import json
# import boto3
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
# from datetime import datetime


class ArticleExtraction:
    def __init__(self, bucket_name: str, parquet_filepath: str, batch_size: int):
        self.bucket_name = bucket_name
        self.parquet_filepath = parquet_filepath
        self.batch_size = batch_size
        # self.date = datetime.today()
        # self.s3 = boto3.client("s3",
        #                        region_name='eu-west-2',
        #                        aws_access_key_id=ACCESS_KEY,
        #                        aws_secret_access_key=SECRET_ACCESS_KEY)
        # print(self.s3.list_buckets())
        # self.article_paths = self.s3.list_objects(Bucket=self.bucket)
        # print(self.article_paths)
        self.spark = SparkSession.builder \
            .appName("ArticleParquetToDF") \
            .getOrCreate()

    def _create_path_dict(self):
        response = self.s3.list_objects(Bucket=self.bucket)
        request_files = response["Contents"]
        path_dict = {}
        for file in request_files:
            filepath = file['Key']
            date = os.path.split(filepath)[0]
            path_dict.setdefault(date, []).append(filepath)
        return path_dict

    def _find_earliest_date(self, path_dict):
        earliest_date = min(set(path_dict.keys()))
        return str(earliest_date)

    def s3_parquet(self):
        return self.spark.read.parquet(f"s3a://{self.bucket_name}/{self.parquet_filepath}") \
            .filter(F.column('language') == 'en')

    # def read_earliest_articles(self):
    #     earliest_date = self.find_earliest_date()
    #     for article in self.path_dict:
    #         obj = self.s3.get_object(Bucket=self.bucket, Key=file[earliest_date])

    def read_earliest_articles(self):
        path_dict = self._create_path_dict()
        earliest_date = self._find_earliest_date(path_dict)
        headlines = []
        for filepath in path_dict[earliest_date]:
            obj = self.s3.get_object(Bucket=self.bucket, Key=filepath)
            article = obj['Body'].read().decode("utf-8")

            with open('test.json', 'w') as f:
                f.write(article)
            try:
                article = json.loads(article)
                headlines.append(article['title'])
            except json.decoder.JSONDecodeError:
                print(article)
        return headlines


if __name__ == '__main__':
    article_extractor = ArticleExtraction()
    print(article_extractor.read_earliest_articles())
