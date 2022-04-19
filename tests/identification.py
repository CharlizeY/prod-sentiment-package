import sparknlp
import unittest
from pyspark.sql import DataFrame
from pyspark.sql.functions import lit

# Relative import
from brand_sentiment.identification import BrandIdentification

import sys
sys.path.append('F:\IMPERIAL MATERIAL\4. Group Project\prod-sentiment-package')


class TestBrandIdentification(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        """ Include useful things for all test cases - keep minimal
        """
        super().__init__(*args, **kwargs)
        self.spark = sparknlp.start()
        self.identifier = BrandIdentification("ner_dl_bert")
        self.fake_data = [{"text": "", "source_domain": "BBC", "date_publish": "2022-04-12T20:25:19", "language": "en"}]
        self.fake_df = self.spark.createDataFrame(self.fake_data)
        # self.fake_df.show()

    def test_output_type(self):
        """ Check if the "predict_brand" function outputs a spark dataframe.
        """
        df_combined = self.identifier.predict_brand(self.fake_df)
        self.assertTrue(isinstance(df_combined, DataFrame))


    def test_predict_empty_df(self):
        """ Check if an empty spark df is returned with an empty "text" column.
        """
        df_combined = self.identifier.predict_brand(self.fake_df)
        df_combined.show()
        self.assertTrue(df_combined.rdd.isEmpty())
        # self.assertTrue(df_combined.subtract(empty_df).rdd.isEmpty())


    def test_predict_df_with_examples(self):
        """ Check if the correct entities and types are returned for several examples.
        """
        self.new_fake_df = self.fake_df.withColumn("text", lit("Google went bankrupt today.")) # [“The Trade Desk", "ORG"]
        self.new_data = [{"text": 'The global unemployment rate rises.', "source_domain": "BBC", "date_publish": "2022-04-06T10:25:19", "language": "en"}, # Will be filtered
        {"text": 'Tesla offered to buy Google based in the US.', "source_domain": "BBC", "date_publish": "2022-03-06T10:25:20", "language": "en"}, 
        {"text": "The lastest work by Andy Warhol was sold by Sotheby's in the US.", "source_domain": "BBC", "date_publish": "2021-02-06T10:25:20", "language": "en"}
        ]
        extra_df = self.spark.createDataFrame(self.new_data)
        self.new_fake_df = self.new_fake_df.union(extra_df)
        df_combined = self.identifier.predict_brand(self.new_fake_df)       
        
        # print(type(df_combined.select(["Predicted_Entity"]).collect()[0][0]))
        print(df_combined.select(["Predicted_Entity"]).collect()[1][0])
        print(df_combined.select(["Predicted_Entity"]).collect()[2][0])
        # The second row should be filtered out (as no entities), so only three rows in the output
        self.assertTrue(df_combined.select(["Predicted_Entity"]).collect()[0][0] == [["Google", "ORG"]])
        self.assertTrue(df_combined.select(["Predicted_Entity"]).collect()[1][0] == [["Tesla", "ORG"], ["Google", "ORG"], ["US", "LOC"]])
        self.assertTrue(df_combined.select(["Predicted_Entity"]).collect()[2][0] == [["Andy Warhol", "PER"], ["Sotheby's", "ORG"], ["US", "LOC"]])


if __name__ == "__main__":
    unittest.main()
