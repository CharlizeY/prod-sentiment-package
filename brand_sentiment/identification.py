import sparknlp
from pyspark.ml import Pipeline
from pyspark.sql.types import StringType, ArrayType
import pyspark.sql.functions as F
# from sparknlp.annotator import *
# from sparknlp.base import *
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Tokenizer, XlnetForTokenClassification, BertEmbeddings, NerDLModel, NerConverter


# The spark udf function that has to be defined outside the class
def get_brand(row_list):
    if not row_list:  # If the list of detected entities is empty
        return []  # Return an empty list

    else:
        # Create a list of lists with entity and type
        data = [[row.result, row.metadata['entity']] for row in row_list]
        return data


class BrandIdentification:
    
    def __init__(self, spark, MODEL_NAME):
        """Creates a class for brand identication using specified model.

        Args:
          MODEL_NAME: Name of the Spark NLP pretrained pipeline.
        """
        self.spark = spark
        self.MODEL_NAME = MODEL_NAME

        # Define Spark NLP pipeline
        documentAssembler = DocumentAssembler() \
            .setInputCol('text') \
            .setOutputCol('document')

        tokenizer = Tokenizer() \
            .setInputCols(['document']) \
            .setOutputCol('token')

        ner_converter = NerConverter() \
            .setInputCols(['document', 'token', 'ner']) \
            .setOutputCol('ner_chunk')
       
        if self.MODEL_NAME == "xlnet_base":
            tokenClassifier = XlnetForTokenClassification \
                .pretrained('xlnet_base_token_classifier_conll03', 'en') \
                .setInputCols(['token', 'document']) \
                .setOutputCol('ner') \
                .setCaseSensitive(True) \
                .setMaxSentenceLength(512)

            nlp_pipeline = Pipeline(stages=[
                documentAssembler, 
                tokenizer,
                tokenClassifier,
                ner_converter
            ])

        elif self.MODEL_NAME == "ner_conll_bert_base_cased":
            # Bert model uses Bert embeddings
            embeddings = BertEmbeddings.pretrained(name='bert_base_cased', lang='en') \
                .setInputCols(['document', 'token']) \
                .setOutputCol('embeddings')

            ner_model = NerDLModel.pretrained(MODEL_NAME, 'en') \
                .setInputCols(['document', 'token', 'embeddings']) \
                .setOutputCol('ner')

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


    def predict_brand(self, df):  # df is a spark df with a column named "text" - headlines or sentences
        # Run the pipeline for the spark df
        df_spark = self.pipeline_model.transform(df)

        # Improve speed of identification using Spark User-defined function
        pred_brand = F.udf(lambda z: get_brand(z), ArrayType(ArrayType(StringType())))  # Output a list of lists containing [entity, type] pairs

        df_spark_combined = df_spark.withColumn("Predicted_Entity", pred_brand('ner_chunk'))
        df_spark_combined = df_spark_combined.select("text", "source_domain", "date_publish", "language", "Predicted_Entity")
        # IF DATE_PUBLISH == NONE: DATE_PUBLISH = EXTRACTION_DATE
        # df_spark_combined.show(100)

        # Remove all rows with no brands detected
        # Only keep lists with at least one identified entity
        df_spark_combined = df_spark_combined.filter(F.size(df_spark_combined.Predicted_Entity) > 0)
        df_spark_combined.show()

        return df_spark_combined
