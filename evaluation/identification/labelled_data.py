# cd to the prod-sentiment-package
# and run the following : python3 -m evaluation.identification.labelled_data

import time
import pandas as pd
import numpy as np
import sparknlp
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from tabulate import tabulate

from brand_sentiment.identification import BrandIdentification

spark = sparknlp.start() # spark = sparknlp.start(gpu=True)

# Load the labelled data with true sentence index, entity name and type
cols_to_keep = ["true_index", "true_entity_name", "true_entity_type"]
true_df_pd = pd.read_csv("./evaluation/identification/entity_labeled.csv", usecols=cols_to_keep) # Use Pandas df for evaluation

# Load the sentence data and keep the first labelled 50 sentences
input_df = spark.read.csv("./evaluation/identification/labeled_entity_sent.csv", header=True) # Use Spark df for modelling
input_df = input_df.select(F.col("text")).limit(50)

# Run the model pipeline
MODEL_NAME = "ner_conll_bert_base_cased"
brand_identifier = BrandIdentification(spark, MODEL_NAME)
predictions = brand_identifier.pipeline_model.transform(input_df) 

# Add a sentence index column to the 'predictions' df
w = Window.orderBy(F.monotonically_increasing_id())
predictions = predictions.withColumn("row_index", F.row_number().over(w)-1) # The index starts from 0
predictions = predictions.withColumn("repeat_number", F.size(F.col("ner_chunk")))

# Create a list of sentence indexes with correct repetitions
row_index_list = predictions.select('row_index').rdd.flatMap(lambda x: x).collect() # Convert a spark col to a list
repeat_number_list = predictions.select('repeat_number').rdd.flatMap(lambda x: x).collect()
final_index_list = [index for i, index in enumerate(row_index_list) for j in range(repeat_number_list[i])] # Make the correct repetitions
# print(final_index_list)

# Create the table with entity-level NER results ('entity' and 'entity_type')
pred_entity_df = predictions.select(F.explode(F.arrays_zip('ner_chunk.result', "ner_chunk.metadata")).alias("cols")).select(\
                                    F.expr("cols['0']").alias("entity_name"),
                                    F.expr("cols['1'].entity").alias('entity_type'))

# Add the 'sentence_index' column to the table with the correct repetitions 
pred_entity_df = pred_entity_df.repartition(1).withColumn("sentence_index", F.udf(lambda id: final_index_list[id])(F.monotonically_increasing_id()))
# pred_entity_df.show(100)

# Concatenate two Pandas df to find matched pairs
pred_entity_df_pd = pred_entity_df.toPandas()
concat_df_pd = pd.concat([true_df_pd, pred_entity_df_pd], axis=1)
concat_df_pd['true_index'] = concat_df_pd['true_index'].fillna(-1).astype('Int64') # Convert the "true_index" from float to int
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(concat_df_pd)
concat_df_pd.dtypes

list_of_true_tuples = zip(concat_df_pd["true_index"], concat_df_pd["true_entity_name"], concat_df_pd["true_entity_type"])
list_of_predicted_tuples = zip(concat_df_pd["sentence_index"], concat_df_pd["entity_name"], concat_df_pd["entity_type"])
condition = [(s, e, t) in list_of_true_tuples for (s, e, t) in list_of_predicted_tuples]
concat_df_pd["matched_or_not"] = np.where(condition, True, False)
# Create dict where each key is tuple -> (true_index, true_entity_name, true_entity_type) with 'True' as the value
# d = dict(zip(((s, e, t) for s, e, t in zip(concat_df_pd["true_index"], concat_df_pd["true_entity_name"], concat_df_pd["true_entity_type"])), "True"))
# # Do a lookup in d using a tuple -> (sentence_index, entity_name, entity_type) and find the matched pairs of entities
# concat_df_pd["matched_or_not"] = "False"
# concat_df_pd["matched_or_not"] = [d[(s, e, t)] for (s, e, t) in zip(concat_df_pd["sentence_index"], concat_df_pd["entity_name"], concat_df_pd["entity_type"])]
print(concat_df_pd)

# Compute the metrics
num_of_true_entities = true_df_pd.shape[0]
num_of_predicted_entities = pred_entity_df_pd.shape[0]
num_of_correct_entities = concat_df_pd[concat_df_pd.matched_or_not == "True"].shape[0]

print(num_of_true_entities)
print(num_of_predicted_entities)
print(num_of_correct_entities)

precision = num_of_correct_entities/num_of_predicted_entities 
recall = num_of_correct_entities/num_of_true_entities 
f1 = 2*precision*recall/(precision+recall)

print(f'precision: {"{:.2f}".format(precision)}, recall: {"{:.2f}".format(recall)}, f1: {"{:.2f}".format(f1)}')
