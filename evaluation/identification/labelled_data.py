import time
import sparknlp
import pyspark.sql.functions as F
from sparknlp.training import CoNLL
from tabulate import tabulate

from brand_sentiment.identification import BrandIdentification

spark = sparknlp.start() # spark = sparknlp.start(gpu=True)

# Load the test data
test_data = CoNLL().readDataset(spark, './CoNLL03.eng.testb')

start_download = time.time()

# Download the pre-trained model
MODEL_NAME = "ner_conll_distilbert_base_cased"
brand_identifier = BrandIdentification(MODEL_NAME)
end_download = time.time()

# Apply the model pipeline on the test set
predictions = brand_identifier.pipeline_model.transform(test_data) 
end_transform = time.time()

# Create the table with entity-level NER results ('entity' and 'entity_type')
pred_entity_df = predictions.select(F.explode(F.arrays_zip('ner_chunk.result',"ner_chunk.metadata")).alias("cols")).select(\
                                    F.expr("cols['0']").alias("entity"),
                                    F.expr("cols['1'].entity").alias('entity_type'))