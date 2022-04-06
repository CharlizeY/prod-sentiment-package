import sparknlp
import unittest
import os

# # Relative import
# from ..brand_sentiment.extraction import ArticleExtraction

import sys
sys.path.append('F:\IMPERIAL MATERIAL\4. Group Project\prod-sentiment-package')
import extraction


class TestArticleExtraction(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        """ Include useful things for all test cases - keep minimal
        """
        super().__init__(*args, **kwargs)
        self.spark = sparknlp.start()
        self.extractor = ArticleExtraction()


    def test_importing_one_article(self, filepath):
        """ Check that an article is imported.
        """
        article = self.extractor.import_one_article(filepath)
        self.assertIsNotNone(article)


    def test_loading_article_sentences(self, filepath):
        """ Check that at least one sentence is extracted from an article body text.
        """
        article = self.extractor.import_one_article(filepath)
        sentences = self.extractor.article_to_sentences(article)
        self.assertNotEqual(0, len(sentences))


    def test_loading_importing_one_headlines(self, filepath):
        """ Check that at a headline is extracted from an article.
        """        
        headline = self.extractor.import_one_headline_json(filepath)
        self.assertIsNotNone(headline)


    def test_import_folder_headlines(self, folderpath):
        """ Check that the number of extracted headlines from the folder
        matches the number of files in it.
        """
        headlines = import_folder_headlines(folderpath)
        num_files = len([1 for x in list(os.scandir(folderpath)) if x.is_file()])
        self.assertEqual(len(headlines), num_files)


if __name__ == "__main__":
    unittest.main()
