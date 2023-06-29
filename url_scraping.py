import requests
from bs4 import BeautifulSoup

def scrape_url(url, output_file):
    print(f"Scraping {url}...")
    # Get the HTML content from the URL
    response = requests.get(url)
    # 200 infers able to scrape essentially
    if response.status_code == 200:
        print(f"Successfully fetched {url}!")
    else:
        print(f"Failed to fetch {url}. Status code: {response.status_code}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the text from the HTML
    text = soup.get_text()

    # this formats the text to cut it up based on periods (ideally the end of a sentence)
    clean_text = text.split(".")
    print(clean_text)
    # Write the text to a file
    file = open(output_file, 'w', encoding="utf-8")
    for sentence in clean_text:
        print(sentence)
        file.write(sentence)
        file.write("\n")

    print(f"Scraping completed. Output file: {output_file}")
    file.close()

# Usage:
url = "https://mbasic.facebook.com/privacy/policy/printable/"
output_file = 'demo_docs/instagram_tos.txt'
scrape_url(url, output_file)
