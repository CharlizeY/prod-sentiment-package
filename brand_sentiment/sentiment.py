import sparknlp
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Tokenizer, BertForSequenceClassification
from sparknlp.pretrained import PretrainedPipeline

# Import functions to manipulate dataframe
from pyspark.sql.functions import array_join
from pyspark.sql.functions import col, explode, expr, greatest
from pyspark.sql.window import Window
from pyspark.sql.functions import monotonically_increasing_id, row_number
from pyspark import SparkFiles
from pyspark.sql.types import StringType, ArrayType
import pyspark.sql.functions as F

from IPython.display import display


# Define the spark udf function outside the class
def append_sentiment(pair_list, sentiment):
        """Append sentiment to each entry in pred brand list. """

        for pair in pair_list:
            pair.append(sentiment)

        return pair_list


class SentimentIdentification:

    def __init__(self, MODEL_NAME):
        """Creates a class for sentiment identication using specified model.

        Args:
          MODEL_NAME: Name of the Spark NLP pretrained pipeline.
        """

        # Create the pipeline instance
        self.MODEL_NAME = MODEL_NAME
        spark = sparknlp.start()

        # Create a custom pipline if requested
        if self.MODEL_NAME == "custom_pipeline":  # https://nlp.johnsnowlabs.com/2021/11/03/bert_sequence_classifier_finbert_en.html
            document_assembler = DocumentAssembler() \
                .setInputCol('title') \
                .setOutputCol('document')

            tokenizer = Tokenizer() \
                .setInputCols(['document']) \
                .setOutputCol('token')

            sequenceClassifier = BertForSequenceClassification \
                  .pretrained('bert_sequence_classifier_finbert', 'en') \
                  .setInputCols(['token', 'document']) \
                  .setOutputCol('class') \
                  .setCaseSensitive(True) \
                  .setMaxSentenceLength(512)

            pipeline = Pipeline(stages=[
                document_assembler,
                tokenizer,
                sequenceClassifier
            ])

            self.pipeline_model = pipeline.fit(spark.createDataFrame([['']]).toDF("title"))

        else:
            self.pipeline_model = PretrainedPipeline(self.MODEL_NAME, lang = 'en')


    def predict_dataframe(self, df):
        """Annotates the input dataframe with the classification results.

        Args:
          df : Pandas or Spark dataframe to classify (must contain a "title" column)
        """
        spark = sparknlp.start()

        if isinstance(df, pd.DataFrame):
            # Convert to spark dataframe for faster prediction
            df_spark = spark.createDataFrame(df)
        else:
            df_spark = df

        # Annotate dataframe with classification results
        df_spark = self.pipeline_model.transform(df_spark)

        # Extract sentiment score
        if self.MODEL_NAME == "custom_pipeline":
            df_spark_scores = df_spark.select(explode(col("class.metadata")).alias("metadata")).select(col("metadata")["Some(positive)"].alias("positive"),
                                                                                            col("metadata")["Some(neutral)"].alias("neutral"),
                                                                                            col("metadata")["Some(negative)"].alias("negative"))
        else:
            df_spark_scores = df_spark.select(explode(col("class.metadata")).alias("metadata")).select(col("metadata")["positive"].alias("positive"),
                                                                                            col("metadata")["neutral"].alias("neutral"),
                                                                                            col("metadata")["negative"].alias("negative"))
        
        df_spark_scores = df_spark_scores.withColumn("score", col("positive")-col("negative"))

        # Extract only target and label columns
        df_spark = df_spark.select("text", "source_domain", "date_publish", "language", "Predicted_Entity", "class.result") # This is to run main.py

        # Rename to result column to Predicted Sentiment
        df_spark = df_spark.withColumnRenamed("result", "Predicted_Sentiment")

        # Convert sentiment from a list to a string
        df_spark = df_spark.withColumn("Predicted_Sentiment", array_join("Predicted_Sentiment", ""))

        # Join the predictions dataframe to the scores dataframe
        # Add temporary column index to join
        w = Window.orderBy(monotonically_increasing_id())
        df_spark_with_index = df_spark.withColumn("columnindex", row_number().over(w))
        df_spark_scores_with_index = df_spark_scores.withColumn("columnindex", row_number().over(w))

        # Join the predictions and the scores in one dataframe
        df_spark_with_index = df_spark_with_index.join(df_spark_scores_with_index,
                                   df_spark_with_index.columnindex == df_spark_scores_with_index.columnindex,
                                   'inner').drop(df_spark_scores_with_index.columnindex)

        # Remove the index column
        df_spark_combined = df_spark_with_index.drop(df_spark_with_index.columnindex)

        # Append sentiment to each entry in pred brand list
        append_sent = F.udf(lambda x, y: append_sentiment(x, y), ArrayType(ArrayType(StringType()))) # Output a list of lists
        df_spark_combined = df_spark_combined.withColumn('Predicted_Entity_and_Sentiment', append_sent('Predicted_Entity', 'Predicted_Sentiment'))

        # Keep positive/neutral/negative probabilities in the output spark df
        df_spark_combined = df_spark_combined.drop('Predicted_Entity', 'Predicted_Sentiment')

        return df_spark_combined