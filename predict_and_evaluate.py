import sparknlp
import logging
from pyspark.sql import SparkSession

from os import environ

from brand_sentiment.sentiment import SentimentIdentification
from pyspark.sql.functions import array_join
from sklearn.metrics import classification_report, accuracy_score

import pandas as pd
from IPython.display import display

# # Store data in a Pandas Dataframe
cols_to_read = ['text',"sentiment (Max's take)"]
df_pandas = pd.read_csv("./input_data.csv", usecols=cols_to_read)

# Rename sentiment to True_Sentiment
df_pandas.rename(columns={"sentiment (Max's take)":"True_Sentiment"},inplace=True)

num_sentences = 500 # Take only the first n labelled sentences
total_num_sentences = df_pandas.shape[0]
df_pandas.drop(df_pandas.index[num_sentences:total_num_sentences], inplace=True)

# Replace 1, 2 , 3 with negative, neutral, positive
df_pandas["True_Sentiment"].replace({1.0: "negative", 2.0: "neutral", 3.0: "positive"}, inplace=True)

# Start spark
spark = sparknlp.start()

# Create sentiment identifier object
identifier_pretrained = SentimentIdentification(spark = spark, MODEL_NAME = "custom_pipeline")

# Convert to spark for transform
df_spark = spark.createDataFrame(df_pandas)

# Annotate dataframe with classification results
df_spark = identifier_pretrained.pipeline_model.transform(df_spark)

# Only keep necessary columns
df_spark = df_spark.select("text", "class.result", "True_Sentiment")
                            
# Rename to result column to Predicted Sentiment
df_spark = df_spark.withColumnRenamed("result", "Predicted_Sentiment")

# Convert sentiment from a list to a string
df_spark = df_spark.withColumn("Predicted_Sentiment", array_join("Predicted_Sentiment", ""))

# Convert to pandas to use sklearn functions
df_pandas_postprocessed = df_spark.toPandas()

display(df_pandas_postprocessed)

# Compute the accuracy
accuracy = accuracy_score(df_pandas_postprocessed["True_Sentiment"], df_pandas_postprocessed["Predicted_Sentiment"])
accuracy *= 100
report = classification_report(df_pandas_postprocessed["True_Sentiment"], df_pandas_postprocessed["Predicted_Sentiment"])

print(f"Accuracy: {accuracy}")
print(report)

# Write postprocessed dataframe to csv file
df_pandas_postprocessed.to_csv('./output_data.csv') 
