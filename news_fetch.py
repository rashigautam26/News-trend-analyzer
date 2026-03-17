# import requests
# import pandas as pd

# API_KEY = "ab1a6f74fd624e5f8ff14a79fc6ce84a"

# url = "https://newsapi.org/v2/top-headlines"

# params = {
#     "sources": "bbc-news",
#     "apiKey": API_KEY
# }

# print("Fetching news from API...")

# response = requests.get(url, params=params)
# data = response.json()

# articles = data["articles"]

# print("Fetched", len(articles), "articles")

# news_list = []

# for article in articles:
#     news_list.append({
#         "title": article["title"],
#         "source": article["source"]["name"],
#         "url": article["url"]
#     })

# df = pd.DataFrame(news_list)

# df.to_csv("news_raw.csv", index=False)

# print("Raw data saved to news_raw.csv")
import requests
import pandas as pd

API_KEY = "ab1a6f74fd624e5f8ff14a79fc6ce84a"

url = "https://newsapi.org/v2/everything"

params = {
    "q": "India",
    "language": "en",
    "sortBy": "publishedAt",
    "pageSize": 50,
    "apiKey": API_KEY
}

print("Fetching news from API...")

response = requests.get(url, params=params)
data = response.json()

# API error check
if data["status"] != "ok":
    print("API Error:", data)
    exit()

articles = data["articles"]

print("Fetched", len(articles), "articles")

news_list = []

for article in articles:
    news_list.append({
        "title": article["title"],
        "source": article["source"]["name"],
        "publishedAt": article["publishedAt"],
        "url": article["url"]
    })

df = pd.DataFrame(news_list)

df.to_csv("news_raw.csv", index=False)

print("Raw data saved to news_raw.csv")