# Scraper Layer Documentation

The Scraper layer is responsible for ingesting real-time maritime news articles from MarineTraffic. This document outlines the architecture, logic, and operational constraints embedded within the scraping pipeline.

The Scraper Later is responsible for ingesting real time maritime news data from MarineTraffic as specified in the Task Description.

# 1. Core Logic 
As per my observation of the MarineTraffic Webpage, i noticed in the developer tools that there was only one API call which was fetching all of the news article data. After some time looking into the API call, i was able to realise that the Category_ID inside the API call was an optional input. After some hit and trials with the pagination inside the API, i was able to extract the latest 100 articles in one API call. All of this was done whilst respecting the robots.txt of the webpage. 

## 1. Handling Pagination
MarineTraffic limits each GraphQL request to a certain number of articles (20 articles per page). The scraper handles pagination systematically to fetch a large volume of articles:

*   **Logic:** The `MarineTrafficClient.get_latest_articles` method accepts `page` and `page_size` parameters.
*   **Implementation:** The main execution block calculates the total iterations required based on the `target_count` and `articles_per_page`. A for loop iterates through `range(1, (target_count // articles_per_page) + 1)`, incrementing the page_num in each request until the desired target count is reached or no more articles are returned.

## 2. Handling HTML Variability


*   **Approach:** We completely bypass HTML scraping. Instead, we reverse-engineered the website's data fetching mechanism and discovered its internal GraphQL API (`https://news.marinetraffic.com/graphql`).
*   **Benefit:** By interacting directly with the GraphQL endpoint, we receive strictly structured, predictable JSON responses. This guarantees 100% resilience against front-end UI redesigns and HTML class name changes, vastly improving pipeline stability.

## 3. Anti-Bot & Rate-Limiting Logic
To assure the scraper functions reliably without triggering cloud defenses (like Cloudflare), basic spoofing and rate-limiting elements are employed:

*   **Header Spoofing:** We inject realistic browser headers to mimic human traffic:
    ```python
    self.headers = {
        "Content-Type": "application/json",
        "Origin": "https://www.marinetraffic.com",
        "Referer": "https://www.marinetraffic.com/",
        "User-Agent": "Mozilla/5.0"
    }
    ```

## 4. Respecting `robots.txt` Principles
Before scraping, we analyzed MarineTraffic's `robots.txt` policies. The relevant section dictates:

```text
User-agent: *
Content-Signal: search=yes,ai-train=no
Allow: /
```

*   **Interpretation:** The website explicitly allows crawling for search indexing purposes (`Allow: /` and `search=yes`), but forbids using the data to train AI models from scratch (`ai-train=no`).      

*   **Compliance Strategy:** This pipeline does not use this data to fine-tune our LLMs. The LLM I am using (LLAMA 4) has frozen weights. We only use the scraped data as retrieved context for generation for the RAG. 
