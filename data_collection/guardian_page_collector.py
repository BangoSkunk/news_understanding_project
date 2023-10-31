import requests
import datetime
from bs4 import BeautifulSoup
import pandas as pd
import os
import argparse
import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=str, default='1980-01-01')
    parser.add_argument('--end-date', type=str, default='2020-03-02')
    parser.add_argument('--part', type=int, default=1)
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--api-key', type=str, default='a48be119-a9d1-497e-a595-b1f37dea391c')
    args = parser.parse_args()
    return args


def create_params(api_key: str,
                  start_date_str: str,
                  end_date_str: str,
                  page_num: int) -> dict:
    return {
        'api-key': api_key,
        'format': 'json',
        'from-date': start_date_str,
        'to-date': end_date_str,
        'page-size': 200,
        'page': page_num
    }


def collect_results_csv(result: requests.models.Response) -> pd.DataFrame:
    res_json = result.json()
    return pd.DataFrame(res_json['response']['results'])


URL = 'https://content.guardianapis.com/search'


def main(args):
    part = args.part
    while True:
        # each iteration creates a new pandas dataframe
        pages_df = pd.DataFrame()
        # the guardian API allows to parse only 190 pages per date snippet
        for page_num in tqdm.tqdm(range(1, 191)):
            params = create_params(api_key=args.api_key,
                                   start_date_str=args.start_date,
                                   end_date_str=args.end_date,
                                   page_num=page_num)
            result = requests.get(URL, params=params)
            if result.reason == 'Too Many Requests':
                print('requests limit exceeded')
                break
            if result.ok:
                current_pages_df = collect_results_csv(result)
                pages_df = pd.concat([pages_df, current_pages_df])
                pages_df.to_csv(os.path.join(args.data_path, f'guardian_news_dataset_p{part}.csv'), index=False)
            else:
                print(page_num)
                print(result.reason)
                break
        if page_num == 190:
            # rewriting end date for the next part
            end_date_str = datetime.datetime.strptime(pages_df.webPublicationDate.min(), '%Y-%m-%dT%H:%M:%SZ').strftime(
                '%Y-%m-%d')
            part += 1
        else:
            print(page_num)
            print(result.reason)
            break


if __name__ == '__main__':
    args = parse_args()
    main(args)
