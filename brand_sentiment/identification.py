from brand_sentiment.extraction import ArticleExtraction

import os
import time
import random
import sparknlp
import pandas as pd
from pyspark import SparkFiles
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, ArrayType
import pyspark.sql.functions as F
import sparknlp
from sparknlp.annotator import *
from sparknlp.base import *
from sparknlp.pretrained import PretrainedPipeline
from sparknlp_display import NerVisualizer


# The spark udf function that has to be defined outside the class
def get_brand(row_list):
    if not row_list: # If the list is empty
        return [] # If no entities detected return an empty list

    else:
        # Create a list of lists with the idetified entity and type
        data = [[row.result, row.metadata['entity']] for row in row_list]
        return data

class BrandIdentification:
    def __init__(self, MODEL_NAME):
        self.MODEL_NAME = MODEL_NAME
        spark = sparknlp.start()

        # Define Spark NLP pipeline
        documentAssembler = DocumentAssembler() \
            .setInputCol('text') \
            .setOutputCol('document')

        tokenizer = Tokenizer() \
            .setInputCols(['document']) \
            .setOutputCol('token')

        # ner_dl and onto_100 model are trained with glove_100d, so the embeddings in the pipeline should match
        if (self.MODEL_NAME == "ner_dl") or (self.MODEL_NAME == "onto_100"):
            embeddings = WordEmbeddingsModel.pretrained('glove_100d') \
                .setInputCols(["document", 'token']) \
                .setOutputCol("embeddings")

        # Bert model uses Bert embeddings
        elif self.MODEL_NAME == "ner_dl_bert":
            embeddings = BertEmbeddings.pretrained(name='bert_base_cased', lang='en') \
                .setInputCols(['document', 'token']) \
                .setOutputCol('embeddings')

        ner_model = NerDLModel.pretrained(MODEL_NAME, 'en') \
            .setInputCols(['document', 'token', 'embeddings']) \
            .setOutputCol('ner')

        ner_converter = NerConverter() \
            .setInputCols(['document', 'token', 'ner']) \
            .setOutputCol('ner_chunk')

        nlp_pipeline = Pipeline(stages=[
            documentAssembler,
            tokenizer,
            embeddings,
            ner_model,
            ner_converter
        ])

        # Create the pipeline model
        empty_df = spark.createDataFrame([['']]).toDF('text') # An empty df with column name "text"
        self.pipeline_model = nlp_pipeline.fit(empty_df)


    def predict_brand(self, df): # df is a spark dataframe with a column named "text", which contains the headlines or sentences
        # Run the pipeline for the spark df containing the "text" column
        spark = sparknlp.start()

        df_spark = self.pipeline_model.transform(df)

        # Improve speed of identification using Spark User-defined function
        pred_brand = F.udf(lambda z: get_brand(z), ArrayType(ArrayType(StringType()))) # Output a list of lists containing [entity, type] pairs

        df_spark_combined = df_spark.withColumn("Predicted_Entity", pred_brand('ner_chunk'))
        df_spark_combined = df_spark_combined.select("text", "source_domain", "date_publish", "language", "Predicted_Entity")
        # df_spark_combined.show(100)

        # Remove all rows with no brands detected
        df_spark_combined  = df_spark_combined.filter(F.size(df_spark_combined.Predicted_Entity) > 0) # Only keep lists with at least one identified entity
        # df_spark_final.show(100)

        return df_spark_combined


if __name__ == '__main__':

    ##### Test for a list of strings
    # spark = sparknlp.start()

    MODEL_NAME = "ner_dl_bert"  # MODEL_NAME = "onto_100"
    brand_identifier = BrandIdentification(MODEL_NAME)

    list_of_headlines = ["Bad news for Google", "Tesla went bankrupt today."]

    brands = brand_identifier.predict_brand(list_of_headlines)



    ##### Test for financial headlines
    # Load the data from Github
    NER_url = 'https://raw.githubusercontent.com/Brand-Sentiment-Tracking/python-package/main/data/NER_test_data.csv'

    # Convert csv data to Pandas dataframe
    df_NER = pd.read_csv(NER_url, header=None).head(500) # 'header=None' prevents pandas eating the first row as headers
    df_NER.columns = ['Brand', 'text']

    # Shuffle the DataFrame rows
    # df_NER = df_NER.sample(frac = 1)

    # Make dataset smaller for faster runtime
    num_sentences = 10
    total_num_sentences = df_NER.shape[0]
    df_NER.drop(df_NER.index[num_sentences:total_num_sentences], inplace=True)


    # # Alternatively, create a preprocessed Spark dataframe from csv
    # from pyspark import SparkFiles
    # spark.sparkContext.addFile(NER_url)
    #
    # # Read raw dataframe
    # df_spark_org = spark.read.csv("file://"+SparkFiles.get("NER_test_data.csv"))
    #
    # # Rename columns
    # df_spark_org = df_spark_org.withColumnRenamed("_c0", "Brand").withColumnRenamed("_c1", "text")
    # df_spark_org = df_spark_org.limit(num_sentences)
    #
    #
    # # Predict brand using either the Pandas or Spark dataframe
    # start = time.time()
    # brand_identifier.predict_brand(df_NER)
    # # brand_identifier.predict_brand(df_spark_org)
    # end = time.time()
    #
    # print(f"{end-start} seconds elapsed to create ranked tables for {num_sentences} sentences in a Pandas dataframe.")
    # # print(f"{end-start} seconds elapsed to create ranked tables for {num_sentences} sentences in a Spark dataframe.")
