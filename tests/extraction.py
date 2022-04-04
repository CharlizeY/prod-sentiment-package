import sparknlp
import unittest

from ..brand_sentiment.extraction import ArticleExtraction

# spark = sparknlp.start()
# article_extractor = ArticleExtraction()
# article = article_extractor.import_one_article('data/article.txt')
# print(article)
# sentences = article_extractor.article_to_sentences(article)
# print(sentences)


class TestArticleExtraction(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        """ Include useful things for all test cases - keep minimal
        """
        super().__init__(*args, **kwargs)

        self.spark = sparknlp.start()
        self.extractor = ArticleExtraction()

    def test_importing_one_article(self):
        """ Tests are self contained.
            i.e. only test one function at a time,
                 and include all prerequisite functions.
        """
        article = self.extractor.import_one_article('data/article.txt')

        self.assertIsNotNone(article)

    def test_loading_article_sentences(self):
        article = self.extractor.import_one_article('data/article.txt')
        sentences = self.extractor.article_to_sentences(article)

        self.assertNotEqual(0, len(sentences))


if __name__ == "__main__":
    unittest.main()
