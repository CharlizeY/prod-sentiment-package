import sparknlp
from sparknlp.base import DocumentAssembler, LightPipeline
from sparknlp.annotator import Tokenizer, BertForSequenceClassification
from sparknlp.pretrained import PretrainedPipeline

import pandas as pd
from pyspark.ml import Pipeline
import pyspark.sql.functions as F
from sklearn.metrics import classification_report, accuracy_score

# Import functions to manipulate dataframe
from pyspark.sql.functions import array_join
from pyspark.sql.functions import col, explode
from pyspark.sql.window import Window
from pyspark.sql.functions import monotonically_increasing_id, row_number
from pyspark.sql.types import StringType, ArrayType


# Define the spark udf function outside the class
def append_sentiment(pair_list, sentiment):
    """Append sentiment to each entry in pred brand list."""

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
                .setInputCol('text') \
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

            self.pipeline_model = pipeline.fit(spark.createDataFrame([['']]).toDF("text"))

        else:
            self.pipeline_model = PretrainedPipeline(self.MODEL_NAME, lang='en')

    def predict_dataframe(self, df):
        """Annotates the input dataframe with the classification results.

        Args:
          df : Pandas or Spark dataframe to classify (must contain a "text" column)
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
            df_spark_scores = df_spark.select(explode(col("class.metadata"))
                                              .alias("metadata")) \
                                      .select(col("metadata")["Some(positive)"].alias("positive"),
                                              col("metadata")["Some(neutral)"].alias("neutral"),
                                              col("metadata")["Some(negative)"].alias("negative"))
        else:
            df_spark_scores = df_spark.select(explode(col("class.metadata"))
                                              .alias("metadata")) \
                                      .select(col("metadata")["positive"].alias("positive"),
                                              col("metadata")["neutral"].alias("neutral"),
                                              col("metadata")["negative"].alias("negative"))

        # Extract only target and label columns
        # df_spark = df_spark.select("text", "True_Sentiment", "class.result")
        df_spark = df_spark.select("text", "source_domain", "date_publish", "language", "Predicted_Entity", "class.result")  # This is to run main.py

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
        df_spark_with_index = df_spark_with_index.join(
            df_spark_scores_with_index,
            df_spark_with_index.columnindex == df_spark_scores_with_index.columnindex,
            'inner').drop(df_spark_scores_with_index.columnindex)

        # Remove the index column
        df_spark_combined = df_spark_with_index.drop(df_spark_with_index.columnindex)

        # Append sentiment to each entry in pred brand list
        append_sent = F.udf(lambda x, y: append_sentiment(x, y), ArrayType(ArrayType(StringType())))  # Output a list of lists
        df_spark_combined = df_spark_combined.withColumn('Predicted_Entity_and_Sentiment', append_sent('Predicted_Entity', 'Predicted_Sentiment'))
        # df_spark_combined = df_spark_combined.select("text", "source_domain", "date_publish", "language", "Predicted_Entity_and_Sentiment")
        # If want to keep positive/neutral/negative probabilities in the output spark df
        df_spark_combined = df_spark_combined.drop('Predicted_Entity', 'Predicted_Sentiment')

        # Convert to pandas dataframe for postprocessing (https://towardsdatascience.com/text-classification-in-spark-nlp-with-bert-and-universal-sentence-encoders-e644d618ca32)
        # df_pandas_postprocessed = df_spark_combined.toPandas()

        # df_spark_combined.show(100)

        # return df_pandas_postprocessed
        return df_spark_combined

    def predict_string_list(self, string_list):
        """Predicts sentiment of the input list of strings.

        Args:
          string_list: List of strings to classify.
        """

        # Annotate input text using pretrained model

        if self.MODEL_NAME == "custom_pipeline":
            pipeline_annotator = LightPipeline(self.pipeline_model)  # Convert the pipeline to an annotator
        else:
            pipeline_annotator = self.pipeline_model

        annotations = pipeline_annotator.annotate(string_list)

        return [annotation['class'][0] for annotation in annotations]  # Return the sentiment list of strings

    def compute_accuracy(self, df_pandas_postprocessed):
        """Computes accuracy by comparing labels of input dataframe.

        Args:
          df_pandas_postprocessed: pandas dataframe containing "True_Sentiment" and "Predicted_Sentiment" columns
        """

        # Compute the accuracy
        accuracy = accuracy_score(df_pandas_postprocessed["True_Sentiment"], df_pandas_postprocessed["Predicted_Sentiment"])
        accuracy *= 100
        classification = classification_report(df_pandas_postprocessed["True_Sentiment"], df_pandas_postprocessed["Predicted_Sentiment"])

        # Alternatively if the input is a postprocessed spark dataframe
        # Compute accuracy by comparing each true label with predicted label
        # accuracy = df_spark.filter(df_spark.Predicted_Sentiment == df_spark.True_Sentiment).count()/ num_sentences

        return accuracy, classification
