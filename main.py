import sparknlp
import logging

from brand_sentiment.extraction import ArticleExtraction
from brand_sentiment.sentiment import SentimentIdentification
from brand_sentiment.identification import BrandIdentification

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)

    spark = sparknlp.start()

    logging.warning(f"Running Apache Spark version {spark.version}")
    logging.warning(f"Running Spark NLP version {sparknlp.version()}")

    article_extractor = ArticleExtraction()
    brand_identifier = BrandIdentification("ner_dl_bert")
    sentimentiser = SentimentIdentification("classifierdl_bertwiki_finance_sentiment_pipeline")

    headlines = article_extractor.import_folder_headlines('articles/https_www_bbc_co_uk_news_technology_60126012')
    print(headlines)
    brand_spark_df = brand_identifier.predict_brand(headlines)
    print(brand_spark_df)
    complete_spark_df = sentimentiser.predict_dataframe(brand_spark_df)
    print(complete_spark_df)
