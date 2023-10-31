import requests
import datetime
from bs4 import BeautifulSoup
import pandas as pd
import concurrent.futures
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str)
    args = parser.parse_args()
    return args


def extract_article_text(url: str,
                         max_retries: int = 20) -> str:
    retries = 0
    article_text = ''
    while article_text == '':
        try:
            result = requests.get(url)
            soup = BeautifulSoup(result.text, 'html.parser')
            article_text = list()
            article_text = ' '.join(text_snippet.text for text_snippet in soup.find_all('p'))
        except Exception as e:
            if retries > max_retries:
                break
            print(f'connection problem: {e}')
            retries += 1
    return article_text


def extract_all_texts_from_list(url_list: list) -> dict:
    url_to_text = dict()
    s = datetime.datetime.now()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the tasks for each URL
        for url, text in zip(url_list, list(executor.map(extract_article_text, url_list))):
            url_to_text[url] = text
    print('Time spent on collecting all papers texts: ', (datetime.datetime.now() - s).seconds, ' sec')
    return url_to_text


def main(args):
    dataset_list = os.listdir(args.dataset_dir)
    dataset_list = [os.path.join(args.dataset_dir, dataset_name) for dataset_name in dataset_list if
                    '.csv' in dataset_name]
    for orig_dataset_path in dataset_list:
        final_dataset_path = orig_dataset_path[:orig_dataset_path.find('.csv')] + '_texts.csv'

        df = pd.read_csv(orig_dataset_path)
        df = df[df.type == 'article']
        url_list = df.webUrl.tolist()

        url_to_text = extract_all_texts_from_list(url_list)
        df['text'] = df.webUrl.map(url_to_text)
        df.to_csv(final_dataset_path, index=False)


if __name__ == '__main__':
    args = parse_args()
    main(args)