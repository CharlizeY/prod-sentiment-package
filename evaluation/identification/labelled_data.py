# cd to the prod-sentiment-package
# and run the following : python3 -m evaluation.identification.labelled_data

import time
import sparknlp
import pyspark.sql.functions as F
from sparknlp.training import CoNLL
from tabulate import tabulate

from brand_sentiment.identification import BrandIdentification

spark = sparknlp.start() # spark = sparknlp.start(gpu=True)

# Load the data
df = spark.read.csv("./evaluation/identification/labeled_entity_sent.csv", header=True)
# Only keep the first 50 headlines for evaluation
df = df.select(F.col('text')).limit(50) 

MODEL_NAME = "ner_conll_bert_base_cased"
brand_identifier = BrandIdentification(spark, MODEL_NAME)
predictions = brand_identifier.pipeline_model.transform(df) 

predictions.printSchema()
predictions.show(truncate=False)

# Create the table with entity-level NER results ('entity' and 'entity_type')
pred_entity_df = predictions.select(F.explode(F.arrays_zip('ner_chunk.result',"ner_chunk.metadata")).alias("cols")).select(\
                                    F.expr("cols['0']").alias("entity"),
                                    F.expr("cols['1'].entity").alias('entity_type'))

pred_entity_df.show(100)