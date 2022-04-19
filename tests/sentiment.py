import sparknlp
import unittest
from pyspark.sql import DataFrame
from pyspark.sql.functions import lit

# Relative import
from brand_sentiment.sentiment import SentimentIdentification

import sys
sys.path.append('F:\IMPERIAL MATERIAL\4. Group Project\prod-sentiment-package')

class TestSentimentIdentification(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        """ Include useful things for all test cases - keep minimal
        """
        super().__init__(*args, **kwargs)
        self.spark = sparknlp.start()
        self.sentimentiser = SentimentIdentification("custom_pipeline")
        self.fake_data = [{"text": "Google went bankrupt today.", "source_domain": "BBC", "date_publish": "2022-04-12T20:25:19", "language": "en", "Predicted_Entity":[["Google", "ORG"]]}]
        self.fake_df = self.spark.createDataFrame(self.fake_data)
        # self.fake_df.show()


    def test_output_type(self):
        """ Check if the "predict_dataframe" function outputs a spark dataframe.
        """
        df_combined = self.sentimentiser.predict_dataframe(self.fake_df)
        self.assertTrue(isinstance(df_combined, DataFrame))
        

    def test_predict_correct_sentiment(self):
        """ Check if the correct sentiment is returned for a given headline/sentence.
        """
        new_data = [
        {"text": "Microsoft stock soars to a new high", "source_domain": "BBC", "date_publish": "2021-02-06T10:25:20", "language": "en", "Predicted_Entity":[["Microsoft", "ORG"]]} 
        ]
        extra_df = self.spark.createDataFrame(new_data)
        new_fake_df = self.fake_df.union(extra_df)
        df_combined = self.sentimentiser.predict_dataframe(new_fake_df)       
        # df_combined.show()

        # print(type(df_combined.select(["Predicted_Entity"]).collect()[0][0]))
        # print(df_combined.select(["Predicted_Entity_and_Sentiment"]).collect()[0][0])
        # print(df_combined.select(["Predicted_Entity_and_Sentiment"]).collect()[1][0])

        self.assertTrue(df_combined.select(["Predicted_Entity_and_Sentiment"]).collect()[0][0] == [["Google", "ORG", "negative"]])
        self.assertTrue(df_combined.select(["Predicted_Entity_and_Sentiment"]).collect()[1][0] == [["Microsoft", "ORG", "positive"]])
        
        
    def test_predict_sentiment_for_multiple_entities(self):
            """ Check if the correct sentiment is returned for all entities in a given headline/sentence.
            """
            new_data = [
            {"text": "The lastest work by Andy Warhol was sold by Sotheby's in the US.", "source_domain": "BBC", "date_publish": "2021-02-06T10:25:20", "language": "en", 
            "Predicted_Entity":[["Andy Warhol", "PER"], ["Sotheby's", "ORG"], ["US", "LOC"]]}
            ]
            new_df = self.spark.createDataFrame(new_data)
            df_combined = self.sentimentiser.predict_dataframe(new_df)  
            # df_combined.show()

            # print(df_combined.select(["Predicted_Entity_and_Sentiment"]).collect()[0][0])
            self.assertTrue(df_combined.select(["Predicted_Entity_and_Sentiment"]).collect()[0][0] == [["Andy Warhol", "PER", "neutral"], ["Sotheby's", "ORG", "neutral"], ["US", "LOC", "neutral"]])

    ###### TO DO: check if the three prob columns exist ####


if __name__ == "__main__":
    unittest.main()