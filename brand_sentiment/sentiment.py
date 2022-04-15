from brand_sentiment.extraction import ArticleExtraction

import sparknlp
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
# from tabulate import tabulate
from sparknlp.annotator import *
from sparknlp.base import *
from sparknlp.pretrained import PretrainedPipeline

# Import functions to manipulate dataframe
from pyspark.sql.functions import array_join
from pyspark.sql.functions import col, explode, expr, greatest
from pyspark.sql.window import Window
from pyspark.sql.functions import monotonically_increasing_id, row_number
from pyspark import SparkFiles
from sklearn.metrics import classification_report, accuracy_score

import time
from IPython.display import display


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
        if self.MODEL_NAME == "custom_pipeline": # https://nlp.johnsnowlabs.com/2021/11/03/bert_sequence_classifier_finbert_en.html
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
            self.pipeline_model = PretrainedPipeline(self.MODEL_NAME, lang = 'en')


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
        df_spark_scores = df_spark.select(explode(col("class.metadata")).alias("metadata")).select(col("metadata")["positive"].alias("positive"),
                                                                                            col("metadata")["neutral"].alias("neutral"),
                                                                                            col("metadata")["negative"].alias("negative"))

        # Extract only target and label columns
        # df_spark = df_spark.select("text", "True_Sentiment", "class.result")
        df_spark = df_spark.select("text", "Predicted_Brand", "class.result") # This is to run main.py

        # Rename to result column to Predicted Sentiment
        df_spark = df_spark.withColumnRenamed("result", "Predicted_Sentiment")

        # Convert sentiment from a list to a string
        df_spark = df_spark.withColumn("Predicted_Sentiment", array_join("Predicted_Sentiment", ""))

        # Join the predictions dataframe to the scores dataframe
        # Add temporary column index to join
        w = Window.orderBy(monotonically_increasing_id())
        df_spark_with_index =  df_spark.withColumn("columnindex", row_number().over(w))
        df_spark_scores_with_index =  df_spark_scores.withColumn("columnindex", row_number().over(w))

        # Join the predictions and the scores in one dataframe
        df_spark_with_index = df_spark_with_index.join(df_spark_scores_with_index,
                                df_spark_with_index.columnindex == df_spark_scores_with_index.columnindex,
                                'inner').drop(df_spark_scores_with_index.columnindex)

        # Remove the index column
        df_spark_combined = df_spark_with_index.drop(df_spark_with_index.columnindex)

        # Convert to pandas dataframe for postprocessing (https://towardsdatascience.com/text-classification-in-spark-nlp-with-bert-and-universal-sentence-encoders-e644d618ca32)
        # df_pandas_postprocessed = df_spark_combined.toPandas()

        df_spark_combined.show(100)
        
        # return df_pandas_postprocessed
        return df_spark_combined


    def predict_string_list(self, string_list):
        """Predicts sentiment of the input list of strings.

        Args:
          string_list: List of strings to classify.
        """
 
        # Annotate input text using pretrained model

        if self.MODEL_NAME == "custom_pipeline":
            pipeline_annotator = LightPipeline(self.pipeline_model) # Convert the pipeline to an annotator
        else:
            pipeline_annotator = self.pipeline_model

        annotations =  pipeline_annotator.annotate(string_list)

        return [annotation['class'][0] for annotation in annotations] # Return the sentiment list of strings


    def compute_accuracy(self, df_pandas_postprocessed):
        """Computes accuracy by comparing labels of input dataframe.

        Args:
          df_pandas_postprocessed: pandas dataframe containing "True_Sentiment" and "Predicted_Sentiment" columns
        """
    
        from sklearn.metrics import classification_report, accuracy_score

        # Compute the accuracy
        accuracy = accuracy_score(df_pandas_postprocessed["True_Sentiment"], df_pandas_postprocessed["Predicted_Sentiment"])
        accuracy *= 100
        classification_report = classification_report(df_pandas_postprocessed["True_Sentiment"], df_pandas_postprocessed["Predicted_Sentiment"])

        # Alternatively if the input is a postprocessed spark dataframe
        # Compute accuracy by comparing each true label with predicted label
        # accuracy = df_spark.filter(df_spark.Predicted_Sentiment == df_spark.True_Sentiment).count()/ num_sentences

        return accuracy, classification_report



if __name__ == '__main__':
    # spark = sparknlp.start()
    
    ################## Predict a list of strings  ##############
    article = ["Bad news for Tesla", "Tesla went bankrupt today."]

    identifier_pretrained = SentimentIdentification(MODEL_NAME = "classifierdl_bertwiki_finance_sentiment_pipeline")
    # identifier_pretrained = SentimentIdentification(MODEL_NAME = "custom_pipeline")

    identifier_pretrained.predict_string_list(article)

    

    ################# Load dataframe ############

    # Convert Kaggle data to Pandas dataframe and preprocess

    # sentiment_url = 'https://raw.githubusercontent.com/Brand-Sentiment-Tracking/python-package/main/data/sentiment_test_data.csv'

    # Choose dataframe type
    # dataframe_type = "Spark"

    # if dataframe_type == "Pandas":
            # # Store data in a Pandas Dataframe
            # df_pandas = pd.read_csv(sentiment_url, header=None)

            # # Change column names (pipelines require a "text" column to predict)
            # df_pandas.columns = ['True_Sentiment', 'text']

            # # Shuffle the DataFrame rows
            # # df_pandas = df_pandas.sample(frac = 1)

            # # Make dataset smaller for faster runtime
            # num_sentences = 10
            # total_num_sentences = df_pandas.shape[0]
            # df_pandas.drop(df_pandas.index[num_sentences:total_num_sentences], inplace=True)


            ################ Classify Pandas dataframe #################

            # start = time.time()
            # # df_pandas_postprocessed = identifier_pretrained.predict_dataframe(df_pandas)
            # df_spark_postprocessed = identifier_pretrained.predict_dataframe(df_pandas)
            # end = time.time()

            # print(f"{end-start} seconds elapsed to classify {num_sentences} sentences from Pandas dataframe.")

            # # display(df_pandas_postprocessed)


    # elif dataframe_type == "Spark":
            # # Create a preprocessed Spark dataframe
            # spark.sparkContext.addFile(sentiment_url)

            # # Read raw dataframe
            # df_spark = spark.read.csv("file://"+SparkFiles.get("sentiment_test_data.csv"))

            # # Rename columns
            # df_spark = df_spark.withColumnRenamed("_c0", "True_Sentiment").withColumnRenamed("_c1", "text")
            # num_sentences = 10
            # df_spark = df_spark.limit(num_sentences)


            ################ Classify Spark dataframe #################

            # start = time.time()
            # # df_pandas_postprocessed = identifier_pretrained.predict_dataframe(df_spark)
            # df_spark_postprocessed = identifier_pretrained.predict_dataframe(df_spark)
            # end = time.time()

            # print(f"{end-start} seconds elapsed to classify {num_sentences} sentences from Spark dataframe.")

            # display(df_pandas_postprocessed)


    # Print accuracy metrics
    # accuracy, report = identifier_pretrained.compute_accuracy(df_pandas_postprocessed)
    # print(accuracy)
    # print(report)



# class BrandSentiment:
#     def __init__(self, model_name):
#         self.model_pipeline = PretrainedPipeline('analyze_sentimentdl_glove_imdb', lang='en')

#     def one_sentence_sentiment(self, sentence):
#         """Sentiment of one sentence using pretrained model given to class.

#         Args:
#             sentence (str): Sentence to do sentiment analysis on.

#         Returns:
#             str: The return value. True for success, False otherwise.

#         """
#         return self.model_pipeline.annotate(sentence)

#     def multiple_sentences_sentiments(self, sentences):
#         return [self.one_sentence_sentiment(sentence) for sentence in sentences]

#     def aggregate_sentences_sentiments(self, sentiments):
#         total_sentiment = 0
#         for sentiment in sentiments:
#             if sentiment == 'pos':
#                 total_sentiment += 1
#             elif sentiment == 'neg':
#                 total_sentiment -= 1
#         return total_sentiment/len(sentiments)

#     def analyse_article(self, sentences):
#         sentiments = self.multiple_sentences_sentiments(sentences)
#         return self.aggregate_sentences_sentiments(sentiments)
