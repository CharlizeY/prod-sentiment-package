import sparknlp
import unittest

# # Relative import
# from ..brand_sentiment.identification import BrandIdentification

import sys
sys.path.append('F:\IMPERIAL MATERIAL\4. Group Project\prod-sentiment-package')
import identification


class TestBrandIdentification(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        """ Include useful things for all test cases - keep minimal
        """
        super().__init__(*args, **kwargs)
        self.spark = sparknlp.start()
        self.identifier = BrandIdentification()


if __name__ == "__main__":
    unittest.main()
