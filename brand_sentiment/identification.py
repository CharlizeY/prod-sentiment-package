from pyspark.ml import Pipeline

from pyspark.sql.types import StringType, ArrayType
from pyspark.sql import functions as F

import sparknlp
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Tokenizer, WordEmbeddingsModel, BertEmbeddings, NerDLModel, NerConverter


# The spark udf function that has to be defined outside the class
def get_brand(row_list):
    if not row_list:  # If the list is empty
        return []  # If no entities detected return an empty list

    else:
        # Create a list of lists with the idetified entity and type
        data = [[row.result, row.metadata['entity']] for row in row_list]
        return data


class BrandIdentification:
    def __init__(self, MODEL_NAME):
        self.MODEL_NAME = MODEL_NAME
        self.spark = sparknlp.start()

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
        empty_df = self.spark.createDataFrame([['']]).toDF('text')  # An empty df with column name "text"
        self.pipeline_model = nlp_pipeline.fit(empty_df)

    def predict_brand(self, df):  # df is a spark dataframe with a column named "text", which contains the headlines or sentences
        # Run the pipeline for the spark df containing the "text" column

        df_spark = self.pipeline_model.transform(df)

        # Improve speed of identification using Spark User-defined function
        pred_brand = F.udf(lambda z: get_brand(z), ArrayType(ArrayType(StringType())))  # Output a list of lists containing [entity, type] pairs

        df_spark_combined = df_spark.withColumn("Predicted_Entity", pred_brand('ner_chunk'))
        df_spark_combined = df_spark_combined.select("text", "source_domain", "date_publish", "language", "Predicted_Entity")
        # df_spark_combined.show(100)

        # Remove all rows with no brands detected
        df_spark_combined = df_spark_combined.filter(F.size(df_spark_combined.Predicted_Entity) > 0)  # Only keep lists with at least one identified entity
        # df_spark_final.show(100)

        return df_spark_combined
