import pandas as pd
import os

import json

def main():

    base = './data/'
    data_path = os.path.join(base, 'keiba.csv')

    data_dict = pd.read_csv(data_path)

    keys = ['年', '競馬場コード', '開催回数', '日数', 'レース数', 'コース距離', 'コースタイプ', '天気', '馬場状態', '頭数',	'着順',	'枠', '馬番', '性', '年齢', '斤量', '人気', '単勝オッズ', '馬体重', '増減', '馬名', '騎手']


    name2index = {}
    for na in data_dict['馬名']:
        if na in name2index: continue
        name2index[na] = len(name2index)
    print("vocab(hourse name) size : ", len(name2index))

    

    jockey2index = {}
    for jock in data_dict['騎手']:
        if jock in jockey2index: continue
        jockey2index[jock] = len(jockey2index)
    print("vocab(jockey name) size : ", len(jockey2index))

    with open(os.path.join(base, 'name_dict.json'), mode='w') as f:
        json.dump(name2index, f, ensure_ascii=False)
    
    with open(os.path.join(base, 'jockey_dict.json'), mode='w') as f:
        json.dump(jockey2index, f, ensure_ascii=False)


    

if __name__ == '__main__':
    main()
