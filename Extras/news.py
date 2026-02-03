import newspaper
import feedparser

def scrape_news_from_feed(feed_url):
    articles = []
    feed = feedparser.parse(feed_url)
    for entry in feed.entries:
        # create a newspaper article object
        article = newspaper.Article(entry.link)
        # download and parse the article
        article.download()
        article.parse()
        # extract relevant information
        articles.append({
            'title': article.title,
            'author': article.authors,
            'publish_date': article.publish_date,
            'content': article.text
        })
    return articles

def GetNews():
    # Use an Indian news RSS feed URL
    feed_url = 'https://timesofindia.indiatimes.com/rssfeeds/1221656.cms'
    articles = scrape_news_from_feed(feed_url)
    newsfeed = []
    # print the extracted articles
    for article in articles:
        newsfeed.append([article['title'], article['content']])
    
    print(newsfeed)
    return newsfeed
