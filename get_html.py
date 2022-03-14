import requests
import time
from bs4 import BeautifulSoup

import pickle
import os

import sys
sys.setrecursionlimit(100000)

def check_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p)

def numStr(num):
    if num >= 10:
        return str(num)
    else:
        return '0' + str(num)

object_dir = './objects'
url_dir = './urls'

check_dir(object_dir)
check_dir(url_dir)

headers = {'User-agent': 'Mozilla/5.0'}

Base = 'https://db.netkeiba.com/race/'

place = ['札幌', '函館' , '福島', '新潟', '東京', '中山', '中京', '京都', '阪神', '小倉']

for year in range(2022, 2023):
    print(year)
    for i in range(1, 11):
        print(place[i-1])
        for j in range(1, 11):
            for k in range(1, 11):
                for l in range(1, 13):

                    quel_num = str(year) + numStr(i) + numStr(j) + numStr(k) + numStr(l)
                    url = Base + quel_num

                    time.sleep(1)
                    html = requests.get(url, headers=headers)
                    html.encoding = 'EUC-JP'

                    soup = BeautifulSoup(html.text, 'html.parser')

                    if len(soup.select('.data_intro h1')) == 0:
                        break
                    else: 

                        print(url)

                        save_dir = os.path.join(object_dir, str(year))
                        check_dir(save_dir)
                            
                        filename = os.path.join(object_dir, str(year), quel_num + '.pkl')
                        with open(filename, mode='wb') as f:
                            pickle.dump(html, f)


