# cd to the prod-sentiment-package
# and run the following : python3 -m evaluation.identification.conll03

import time
import sparknlp
import pyspark.sql.functions as F
from sparknlp.training import CoNLL
from sklearn.metrics import classification_report
from tabulate import tabulate

from brand_sentiment.identification import BrandIdentification


class TokenEvaluation:
    def __init__(self, df, true_col_name, pred_col_name):
        self.df = df
        self.true_col_name = true_col_name
        self.pred_col_name = pred_col_name
        
    def count_all_entities(self, col_name):
        num_of_entities = self.df.filter((F.col(col_name) == 'B-LOC') |
                                         (F.col(col_name) == 'B-ORG') |
                                         (F.col(col_name) == 'B-PER') |
                                         (F.col(col_name) == 'B-MISC')).count() 

        return num_of_entities

    def count_all_correct_entities(self):
        num_of_correct_entities = self.df.filter(((F.col(self.true_col_name) == 'B-LOC') |
                                                  (F.col(self.true_col_name) == 'B-ORG') |
                                                  (F.col(self.true_col_name) == 'B-PER') |
                                                  (F.col(self.true_col_name) == 'B-MISC')) &
                                                  (F.col(self.true_col_name) == F.col(self.pred_col_name))).count()

        return num_of_correct_entities

    def count_entities_by_type(self, col_name, entity_type):
        num_of_entities_by_type = self.df.filter(F.col(col_name) == entity_type).count() 

        return  num_of_entities_by_type

    def count_correct_entities_by_type(self, entity_type):
        num_of_correct_entities_by_type = self.df.filter((F.col(self.true_col_name) == entity_type) &
                                                    (F.col(self.true_col_name) == F.col(self.pred_col_name))).count()

        return  num_of_correct_entities_by_type

    def compute_classi_metrics(self, num_of_corrected_entities, num_of_predicted_entities, num_of_true_entities):    
        ''' Compute the precison, recall, F1 score for each entity type.
            Return: prec(float): the precision is the percentage of entities found that are correctly matched;
                    rec(float): the recall is the percentage of entities in the test set that are found;
                    f1(float): the F1 score.
        '''
        prec = num_of_correct_entities/num_of_predicted_entities 
        rec = num_of_correct_entities/num_of_true_entities 
        f1 = 2*prec*rec/(prec+rec)

        return "{:.2f}".format(prec), "{:.2f}".format(rec), "{:.2f}".format(f1)


if __name__ == '__main__':
    spark = sparknlp.start() # spark = sparknlp.start(gpu=True)

    # Load the test data
    test_data = CoNLL().readDataset(spark, './CoNLL03.eng.testb')

    start_download = time.time()

    # Download the pre-trained model
    MODEL_NAME = "ner_conll_bert_base_cased"
    brand_identifier = BrandIdentification(spark, MODEL_NAME)
    end_download = time.time()

    # Apply the model pipeline on the test set
    predictions = brand_identifier.pipeline_model.transform(test_data) 
    end_transform = time.time()

    # Create the spark df with token-level NER results 
    pred_token_df = predictions.select(F.explode(F.arrays_zip('token.result','label.result','ner.result')).alias("cols")) \
                               .select(F.expr("cols['0']").alias("token"),
                                       F.expr("cols['1']").alias("ground_truth"),
                                       F.expr("cols['2']").alias("prediction"))
    # Drop rows with Null values 
    pred_token_df = pred_token_df.na.drop()
    
    # Drop rows with 'O's (non-entity) in the 'ground_truth', so don't produce metrics for the 'O' class
    pred_token_df_no_O = pred_token_df.filter((pred_token_df.ground_truth != 'O'))
    preds_token_pd_df = pred_token_df_no_O.toPandas()
    
    token_evaluator = TokenEvaluation(pred_token_df, 'ground_truth', 'prediction')

    # Count the number of true entities 
    num_of_true_entities =  token_evaluator.count_all_entities('ground_truth') # 5644

    # Count the number of predicted entities
    num_of_predicted_entities =  token_evaluator.count_all_entities('prediction')

    # Count the number of correctly matched entities
    num_of_correct_entities =  token_evaluator.count_all_correct_entities()

    # Count the number of true entities for each entity type
    num_of_true_entities_LOC =  token_evaluator.count_entities_by_type('ground_truth', 'B-LOC')
    num_of_true_entities_ORG =  token_evaluator.count_entities_by_type('ground_truth', 'B-ORG')
    num_of_true_entities_PER =  token_evaluator.count_entities_by_type('ground_truth', 'B-PER')
    num_of_true_entities_MISC =  token_evaluator.count_entities_by_type('ground_truth', 'B-MISC')

    # Count the number of predicted entities for each entity type
    num_of_predicted_entities_LOC =  token_evaluator.count_entities_by_type('prediction', 'B-LOC')
    num_of_predicted_entities_ORG = token_evaluator.count_entities_by_type('prediction', 'B-ORG')
    num_of_predicted_entities_PER = token_evaluator.count_entities_by_type('prediction', 'B-PER')
    num_of_predicted_entities_MISC = token_evaluator.count_entities_by_type('prediction', 'B-MISC')

    # Count the number of correctly matched entities for each entity type
    num_of_correct_entities_LOC = token_evaluator.count_correct_entities_by_type('B-LOC')
    num_of_correct_entities_ORG = token_evaluator.count_correct_entities_by_type('B-ORG')
    num_of_correct_entities_PER = token_evaluator.count_correct_entities_by_type('B-PER')
    num_of_correct_entities_MISC = token_evaluator.count_correct_entities_by_type('B-MISC')

    # Compute the precison, recall, F1 score for each entity type
    prec, rec, f1 = token_evaluator.compute_classi_metrics(num_of_correct_entities, num_of_predicted_entities, num_of_true_entities)
    prec_LOC, rec_LOC, f1_LOC = token_evaluator.compute_classi_metrics(num_of_correct_entities_LOC, num_of_predicted_entities_LOC, num_of_true_entities_LOC)
    prec_ORG, rec_ORG, f1_ORG = token_evaluator.compute_classi_metrics(num_of_correct_entities_ORG, num_of_predicted_entities_ORG, num_of_true_entities_ORG)
    prec_PER, rec_PER, f1_PER = token_evaluator.compute_classi_metrics(num_of_correct_entities_PER, num_of_predicted_entities_PER, num_of_true_entities_PER)
    prec_MISC, rec_MISC, f1_MISC = token_evaluator.compute_classi_metrics(num_of_correct_entities_MISC, num_of_predicted_entities_MISC, num_of_true_entities_MISC)

    metrics = [['All', prec, rec, f1],
    ['LOC', prec_LOC, rec_LOC, f1_LOC],
    ['ORG', prec_ORG, rec_ORG, f1_ORG],
    ['PER', prec_PER, rec_PER, f1_PER],
    ['MISC', prec_MISC, rec_MISC, f1_MISC]] 
    
    # Print the metrics
    print(classification_report(preds_token_pd_df['ground_truth'], preds_token_pd_df['prediction']))
    print(tabulate(metrics, headers=["entity_type", "precision", "recall", "F1"]))

    end_evaluate = time.time()

    # Print the running times
    print(f'The time elasped to download the pre-trained {MODEL_NAME} model is {end_download - start_download} seconds.')
    print(f'The time elasped to apply the pre-trained {MODEL_NAME} model to the test set is {end_transform - end_download} seconds.')
    print(f'The time elasped to evaluate the pre-trained {MODEL_NAME} model is {end_evaluate - end_transform} seconds.')