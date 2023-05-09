import requests
from bs4 import BeautifulSoup

def scrape_url(url, output_file):
    print(f"Scraping {url}...")
    # Get the HTML content from the URL
    response = requests.get(url)
    if response.status_code == 200:
        print(f"Successfully fetched {url}!")
    else:
        print(f"Failed to fetch {url}. Status code: {response.status_code}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the text from the HTML
    text = soup.get_text()

    # Write the text to a file
    with open(output_file, 'w') as file:
        file.write(text)

    print(f"Scraping completed. Output file: {output_file}")

# Example usage:
url = "https://www.betterhelp.com/terms/"
output_file = 'betterhelp_tos.txt'
scrape_url(url, output_file)
