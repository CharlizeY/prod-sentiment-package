import pandas as pd
import numpy as np
import json
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from sparknlp.annotator import *
from sparknlp.base import *
import sparknlp
from sparknlp.pretrained import PretrainedPipeline

from brand_sentiment.extraction import ArticleExtraction
from brand_sentiment.sentiment import SentimentIdentification
from brand_sentiment.identification import BrandIdentification


def test():
    # spark = sparknlp.start()
    article_extractor = ArticleExtraction()
    # brand_identifier = BrandIdentifier()
    # sentimentiser = BrandSentiment()
    list_of_headlines = article_extractor.import_folder_headlines('articles/https_www_bbc_co_uk_news_technology_60126012')

    return list_of_headlines


if __name__ == '__main__':
    spark = sparknlp.start()
    # article_extractor = ArticleExtraction()
    brand_identifier = BrandIdentification("ner_dl_bert")
    sentimentiser = SentimentIdentification(MODEL_NAME = "classifierdl_bertwiki_finance_sentiment_pipeline")
    # article = article_extractor.import_one_article('data/article.txt')
    # print(article)
    # print(test())
    list_of_headlines = test()
    brand_spark_df = brand_identifier.predict_brand(list_of_headlines)
    complete_spark_df = sentimentiser.predict_dataframe(brand_spark_df)
