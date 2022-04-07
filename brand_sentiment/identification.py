import random
import pandas as pd

from pyspark.ml import Pipeline

from pyspark.sql.types import StringType
from pyspark.sql import functions as F

from sparknlp.annotator import *
from sparknlp.base import *


# The spark udf function that has to be defined outside the class
def get_brand(row_list):
    if not row_list: # If the list is empty
        return "None"

    else:
        # Create a pandas df with entity names and types
        data = [[row.result, row.metadata['entity']] for row in row_list]
        df_pd = pd.DataFrame(data, columns = ['Entity', 'Type'])
      
        # Filter only ORGs
        df_pd = df_pd[df_pd["Type"] == "ORG"]

        # Rank the ORGs by frequencies
        ranked_df = df_pd["Entity"].value_counts() # a Pandas Series object
            
        # If no ORG identified in headline, return None
        if len(ranked_df.index) == 0:
           return "None"

        # If only one ORG appears in headline, return it
        elif len(ranked_df.index) == 1:
           return ranked_df.index[0]

        # If one ORG appear more than the others, return that one 
        elif ranked_df[0] > ranked_df[1]:
            return ranked_df.index[0] 

        else: # If multiple ORGs appear the same time, return randomly (TO BE MODIFIED)
            return random.choice([ranked_df.index[0], ranked_df.index[1]])
            # TO DO: break even - Wikidata for article body #


            
class BrandIdentification:
    def __init__(self, spark_session, model="ner_dl_bert"):
        self.spark = spark_session
        self.model = model

        # Define Spark NLP pipeline 
        documentAssembler = DocumentAssembler() \
            .setInputCol('text') \
            .setOutputCol('document')

        tokenizer = Tokenizer() \
            .setInputCols(['document']) \
            .setOutputCol('token')

        # ner_dl and onto_100 model are trained with glove_100d, so the embeddings in the pipeline should match
        if (self.model == "ner_dl") or (self.model == "onto_100"):
            embeddings = WordEmbeddingsModel.pretrained('glove_100d') \
                .setInputCols(["document", 'token']) \
                .setOutputCol("embeddings")

        # Bert model uses Bert embeddings
        elif self.model == "ner_dl_bert":
            embeddings = BertEmbeddings.pretrained(name='bert_base_cased', lang='en') \
                .setInputCols(['document', 'token']) \
                .setOutputCol('embeddings')

        ner_model = NerDLModel.pretrained(model, 'en') \
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
        empty_df = self.spark.createDataFrame([['']]).toDF('text') # An empty df with column name "text"
        self.pipeline_model = nlp_pipeline.fit(empty_df)


    def predict_brand(self, text):
        # text could be a pandas dataframe or a Spark dataframe (both with a column "text"), a list of strings or a single string
        # Run the pipeline for the text
        if isinstance(text, pd.DataFrame): text_df = self.spark.createDataFrame(text) # If input a pandas dataframe
        elif isinstance(text, list): text_df = self.spark.createDataFrame(pd.DataFrame({'text': text})) # If input a list of strings
        elif isinstance(text, str): text_df = self.spark.createDataFrame(pd.DataFrame({'text': text}, index=[0])) # If input a single string
        else: text_df = text

        df_spark = self.pipeline_model.transform(text_df) 

        # Improve speed of identification using Spark User-defined function
        pred_brand = F.udf(lambda z: get_brand(z), StringType()) # Output a string
        # spark.udf.register("pred_brand", pred_brand)

        df_spark_combined = df_spark.withColumn('Predicted_Brand', pred_brand('ner_chunk'))
        df_spark_combined = df_spark_combined.select("text", "Predicted_Brand")
        # df_spark_combined.show(100)

        # Remove all rows with no brands detected
        df_spark_final=df_spark_combined.filter(df_spark_combined.Predicted_Brand != 'None')
        df_spark_final.show(100)

        return df_spark_final