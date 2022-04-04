import json
import re

from newsplease import NewsPlease
#from newspaper import Article

article = NewsPlease.from_url("https://www.bbc.co.uk/news/technology-60126012")
#article = Article("https://www.bbc.co.uk/news/technology-60126012")
article_name = re.sub(r"\W+", "_", article.url)
#article.download()
#article.parse()

with open(f"test_articles/{article_name}", "w") as f:
    json.dump(article.__dict__, f, indent=4, default=str,
              sort_keys=True, ensure_ascii=False)