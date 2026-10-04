import pandas as pd
import httpx
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import csv
import os

csv_file = 'Datasets/credits.csv'
df = pd.read_csv(csv_file)

df = df.drop_duplicates(subset=['person_id'])

actor_names = df['name'].tolist()

actor_ids = df['id'].tolist()

person_ids = df['person_id'].tolist()

num_threads = 500

def get_word_count(actor_name):
    try:
        wikipedia_url = f"https://en.wikipedia.org/wiki/{actor_name.replace(' ', '_')}"

        response = httpx.get(wikipedia_url)
        
        if response.status_code == 404:
            print(f"Wikipedia page not found for {actor_name}")
            return 0
        
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')

        main_content = soup.find(id='mw-content-text')
        
        if main_content:
            words = main_content.text.split()
            return len(words)
        else:
            return 0
    except httpx.RequestError as e:
        print(f"Error fetching data for {actor_name}: {str(e)}")
        return 0

output_csv_file = 'Datasets/credits_with_wiki.csv'

if os.path.exists(output_csv_file):
    existing_df = pd.read_csv(output_csv_file)
    
    last_index = existing_df.index.max()
    
    start_index = last_index + 1
else:
    start_index = 0

with open(output_csv_file, mode='a', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['id', 'person_id', 'name', 'page_size']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    if os.stat(output_csv_file).st_size == 0:
        writer.writeheader()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        def fetch_and_write(i):
            actor_name = actor_names[i]
            word_count = get_word_count(actor_name)
            writer.writerow({
                'id': actor_ids[i],
                'person_id': person_ids[i],
                'name': actor_name,
                'page_size': word_count
            })
        
        futures = [executor.submit(fetch_and_write, i) for i in range(start_index, len(actor_names))]
        for future in futures:
            future.result()


