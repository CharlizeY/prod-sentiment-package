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

<<<<<<< Updated upstream

def compute_num_of_entities(true_df, pred_df, concat_df, entity_type):
    ''' Compute the number of true, predicted, correctly matched entities for each entity type.
        The input dataframes are all Pandas dataframes.
    '''
    num_of_true_entities = true_df[true_df['true_entity_type'] == entity_type].shape[0]
    num_of_predicted_entities = pred_df[pred_df['entity_type'] == entity_type].shape[0]
    num_of_correct_entities = concat_df[(concat_df["matched_or_not"] == "True") & (concat_df["entity_type"] == entity_type)].shape[0]

    return num_of_true_entities, num_of_predicted_entities, num_of_correct_entities
=======
spark = sparknlp.start()  # spark = sparknlp.start(gpu=True)

# Load the data
df = spark.read.csv("./evaluation/identification/labeled_entity_sent.csv", header=True)
# Only keep the first 50 headlines for evaluation
df = df.select(F.col('text')).limit(50)

MODEL_NAME = "ner_conll_bert_base_cased"
brand_identifier = BrandIdentification(spark, MODEL_NAME)
predictions = brand_identifier.pipeline_model.transform(df)
>>>>>>> Stashed changes


def compute_classi_metrics(num_of_correct_entities, num_of_predicted_entities, num_of_true_entities):
    ''' Compute the precison, recall, F1 score for each entity type.
        Return: prec(float): the precision is the percentage of entities found that are correctly matched;
                rec(float): the recall is the percentage of entities in the test set that are found;
                f1(float): the F1 score.
    '''
    prec = num_of_correct_entities/num_of_predicted_entities
    rec = num_of_correct_entities/num_of_true_entities
    f1 = 2*prec*rec/(prec+rec)

<<<<<<< Updated upstream
    return "{:.2f}".format(prec), "{:.2f}".format(rec), "{:.2f}".format(f1)


if __name__ == '__main__':
    spark = sparknlp.start()  # spark = sparknlp.start(gpu=True)

    # Load the labelled data with true sentence index, entity name and type
    cols_to_keep = ["true_index", "true_entity_name", "true_entity_type"]
    true_df_pd = pd.read_csv("./evaluation/identification/entity_labeled.csv", usecols=cols_to_keep)  # Use Pandas df for evaluation

    # Load the sentence data and keep the first labelled 50 sentences
    input_df = spark.read.csv("./evaluation/identification/labeled_entity_sent.csv", header=True)  # Use Spark df for modelling
    input_df = input_df.select(F.col("text")).limit(50)

    # Run the model pipeline
    MODEL_NAME = "ner_conll_bert_base_cased"
    brand_identifier = BrandIdentification(spark, MODEL_NAME)
    predictions = brand_identifier.pipeline_model.transform(input_df)

    # Add a sentence index column to the 'predictions' df
    w = Window.orderBy(F.monotonically_increasing_id())
    predictions = predictions.withColumn("row_index", F.row_number().over(w)-1)  # The index starts from 0
    predictions = predictions.withColumn("repeat_number", F.size(F.col("ner_chunk")))

    # Create a list of sentence indexes with correct repetitions
    row_index_list = predictions.select('row_index').rdd.flatMap(lambda x: x).collect()  # Convert a spark col to a list
    repeat_number_list = predictions.select('repeat_number').rdd.flatMap(lambda x: x).collect()
    final_index_list = [index for i, index in enumerate(row_index_list) for j in range(repeat_number_list[i])]  # Make the correct repetitions

    # Create the table with entity-level NER results ('entity' and 'entity_type')
    pred_df = predictions.select(F.explode(F.arrays_zip('ner_chunk.result', "ner_chunk.metadata")).alias("cols")).select(\
                                        F.expr("cols['0']").alias("entity_name"),
                                        F.expr("cols['1'].entity").alias('entity_type'))

    # Add the 'sentence_index' column to the table with the correct repetitions
    pred_df = pred_df.repartition(1).withColumn("sentence_index", F.udf(lambda id: final_index_list[id])(F.monotonically_increasing_id()))

    # Concatenate two Pandas df
    pred_df_pd = pred_df.toPandas()
    concat_df_pd = pd.concat([true_df_pd, pred_df_pd], axis=1)
    concat_df_pd['true_index'] = concat_df_pd['true_index'].fillna(-1).astype('Int64')  # Convert the "true_index" from float to int
    concat_df_pd['sentence_index'] = pd.to_numeric(concat_df_pd['sentence_index'])  # Convert the "sentence_index" from object to int

    # Search for matched pairs of entities
    list_of_true_tuples = list(zip(concat_df_pd["true_index"], concat_df_pd["true_entity_name"], concat_df_pd["true_entity_type"]))
    list_of_predicted_tuples = list(zip(concat_df_pd["sentence_index"], concat_df_pd["entity_name"], concat_df_pd["entity_type"]))
    condition = [(s, e, t) in list_of_true_tuples for (s, e, t) in list_of_predicted_tuples]
    concat_df_pd["matched_or_not"] = np.where(condition, "True", "False")

    # Compute the metrics
    num_of_true_entities = true_df_pd.shape[0]
    num_of_predicted_entities = pred_df_pd.shape[0]
    num_of_correct_entities = concat_df_pd[concat_df_pd.matched_or_not == "True"].shape[0]

    num_of_true_entities_LOC, num_of_predicted_entities_LOC,num_of_correct_entities_LOC = compute_num_of_entities(true_df_pd, pred_df_pd, concat_df_pd, "LOC")
    num_of_true_entities_ORG, num_of_predicted_entities_ORG,num_of_correct_entities_ORG = compute_num_of_entities(true_df_pd, pred_df_pd, concat_df_pd, "ORG")
    num_of_true_entities_PER, num_of_predicted_entities_PER,num_of_correct_entities_PER = compute_num_of_entities(true_df_pd, pred_df_pd, concat_df_pd, "PER")

    # Compute the precison, recall, F1 score for each entity type
    prec, rec, f1 = compute_classi_metrics(num_of_correct_entities, num_of_predicted_entities, num_of_true_entities)
    prec_LOC, rec_LOC, f1_LOC = compute_classi_metrics(num_of_correct_entities_LOC, num_of_predicted_entities_LOC, num_of_true_entities_LOC)
    prec_ORG, rec_ORG, f1_ORG = compute_classi_metrics(num_of_correct_entities_ORG, num_of_predicted_entities_ORG, num_of_true_entities_ORG)
    prec_PER, rec_PER, f1_PER = compute_classi_metrics(num_of_correct_entities_PER, num_of_predicted_entities_PER, num_of_true_entities_PER)

    metrics = [['All', prec, rec, f1],
               ['ORG', prec_ORG, rec_ORG, f1_ORG],
               ['PER', prec_PER, rec_PER, f1_PER],
               ['LOC', prec_LOC, rec_LOC, f1_LOC]]

    print(tabulate(metrics, headers=["entity_type", "precision", "recall", "F1"]))
=======
pred_entity_df.show(100)
>>>>>>> Stashed changes
