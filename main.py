import sparknlp
import logging

from brand_sentiment.extraction import ArticleExtraction
from brand_sentiment.sentiment import SentimentIdentification
from brand_sentiment.identification import BrandIdentification

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)

    spark = sparknlp.start()
    
    logging.warning(f"Running Apache Spark version {spark.version}")
    logging.warning(f"Running JSL Spark NLP version {sparknlp.version()}")

    article_extractor = ArticleExtraction()
    brand_identifier = BrandIdentification("ner_dl_bert")
    #sentimentiser = SentimentIdentification("classifierdl_bertwiki_finance_sentiment_pipeline")

    #list_of_headlines = list_of_headlines = article_extractor.import_folder_headlines('articles/https_www_bbc_co_uk_news_technology_60126012')
    
    #brand_spark_df = brand_identifier.predict_brand(list_of_headlines)
    #complete_spark_df = sentimentiser.predict_dataframe(brand_spark_df)
