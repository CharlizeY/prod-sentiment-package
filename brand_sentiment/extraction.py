import nltk.data
import os
import json


class ArticleExtraction:
    def __init__(self):
        self.headlines = []

    def import_one_article(self, filepath):

        with open(filepath, "rb") as f:
            article = f.read().decode('utf-8')

        return article

    def import_one_headline_json(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.headlines.append(data['title'])

    def article_to_sentences(self, article):
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        return '\n-----\n'.join(tokenizer.tokenize(article))

    def import_folder_headlines(self, folderpath):
        for filename in os.listdir(folderpath):
            filepath = os.path.join(folderpath, filename)
            self.import_one_headline_json(filepath)
        return self.headlines


if __name__ == '__main__':
    article_extractor = ArticleExtraction()
    article_extractor.import_folder_headlines('./data/cc_download_articles/cyprus-mail.com')
