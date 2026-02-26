import requests
import json
from typing import Dict, Any


class MarineTrafficClient:
    def __init__(self):
        self.url = "https://news.marinetraffic.com/graphql"
        self.headers = {
            "Content-Type": "application/json",
            "Origin": "https://www.marinetraffic.com",
            "Referer": "https://www.marinetraffic.com/",
            "User-Agent": "Mozilla/5.0"
        }

    def get_latest_articles(
        self,
        category_id: str = "34",
        page: int = 1,
        page_size: int = 12,
        skip: int = 0,
        excluded_article_ids=None
    ) -> Dict[str, Any]:

        if excluded_article_ids is None:
            excluded_article_ids = []

        query = """
        query getLatest(
            $categoryId: ID
            $page: Int!
            $pageSize: Int!
            $skip: Int!
            $excludedArticleIds: [ID!]!
        ) {
            latestArticles(
                categoryId: $categoryId
                page: $page
                pageSize: $pageSize
                skip: $skip
                excludedArticleIds: $excludedArticleIds
            ) {
                id
                title
                content
                publishedAt
                slug
                isFeatured
                media {
                    data {
                        attributes {
                            name
                            url
                        }
                    }
                }
                category {
                    data {
                        id
                        attributes {
                            name
                        }
                    }
                }
                author {
                    data {
                        attributes {
                            name
                            image {
                                data {
                                    attributes {
                                        url
                                    }
                                }
                            }
                        }
                    }
                }
                assets {
                    data {
                        id
                        attributes {
                            assetId
                            assetName
                            assetType
                        }
                    }
                }
            }
        }
        """

        payload = {
            "query": query,
            "variables": {
                "categoryId": category_id,
                "page": page,
                "pageSize": page_size,
                "skip": skip,
                "excludedArticleIds": excluded_article_ids
            },
            "operationName": "getLatest"
        }

        response = requests.post(
            self.url,
            headers=self.headers,
            json=payload,
            timeout=15
        )

        response.raise_for_status()  # Raises error if status != 200

        data = response.json()

        if "errors" in data:
            raise Exception(f"GraphQL Error: {data['errors']}")

        return data


if __name__ == "__main__":
    client = MarineTrafficClient()

    try:
        # Setting cat_id to None to fetch across ALL categories
        cat_id = None 
        target_count = 100
        articles_per_page = 20
        total_articles = []
        
        def build_source_url(article: Dict[str, Any]) -> str:
            """Constructs the full MarineTraffic article URL from GraphQL data."""
            try:
                # 1. Get Category Details
                cat_data = article.get('category', {}).get('data', {})
                cat_id = cat_data.get('id', 'unknown')
                cat_name = cat_data.get('attributes', {}).get('name', 'general').lower().replace(' ', '-')
                
                # 2. Get Year from publishedAt (2026-02-25T...)
                pub_date = article.get('publishedAt', '')
                year = pub_date[:4] if pub_date else '2026'
                
                # 3. Get Article ID and Slug
                art_id = article.get('id', '')
                slug = article.get('slug', '')
                
                return f"https://www.marinetraffic.com/en/maritime-news/{cat_id}/{cat_name}/{year}/{art_id}/{slug}"
            except Exception:
                return f"https://www.marinetraffic.com/en/maritime-news/article/{article.get('slug', '')}"

        # We need to loop through pages because the server limits each request to 20 articles
        for page_num in range(1, (target_count // articles_per_page) + 1):
            print(f"Fetching page {page_num}...")
            result = client.get_latest_articles(
                category_id=cat_id,
                page=page_num,
                page_size=articles_per_page,
                skip=0
            )
            
            page_articles = result["data"]["latestArticles"]
            if not page_articles:
                print("No more articles found.")
                break
            
            # Enrich with Source URL
            for art in page_articles:
                art['source_url'] = build_source_url(art)
                
            total_articles.extend(page_articles)
            print(f"Collected {len(total_articles)} articles so far.")

        print(f"\nFinal Count: Fetched {len(total_articles)} articles total.\n")

        for article in total_articles[:5]: # Print first 5 as sample
            print(f"Title: {article['title']}")
            print(f"URL:   {article['source_url']}")
            print("-" * 30)

        # Save all 100 articles to JSON
        with open("latest_articles.json", "w", encoding="utf-8") as f:
            json.dump(total_articles, f, indent=4, ensure_ascii=False)
        print("\nAll articles saved to latest_articles.json")

    except Exception as e:
        print("Error:", e)