import pandas as pd
from tqdm import tqdm

def coursetype2num(x):
    if x == '芝':
        return 1.
    elif x == 'ダート':
        return 2.
    else:
        print(x)

def weather2num(x):
    if x == '晴':
        return 1.
    elif x == '曇':
        return 2.
    elif x == '小雨':
        return 3.
    elif x == '雨':
        return 4. 
    elif x == '小雪':
        return 5.
    elif x == '雪':
        return 6.
    else:
        print(x)

def state2num(x):
    if x == '良':
        return 1.
    elif x == '稍重':
        return 2.
    elif x == '重':
        return 3.
    elif x == '不良':
        return 4.
    else:
        print(x)

def sex2num(x):
    if x == '牡':
        return 1.
    elif x == '牝':
        return 2.
    elif x == 'セ':
        return 3.
    else:
        print(x)

def rank2label(x):
    try:
        if int(x) <= 3:
            return 1.
        else:
            return 0.
    except:
        return 0.
    
def str2num(key, data):
    if key == 'コースタイプ':
        return coursetype2num(data)
    elif key == '天気':
        return weather2num(data)
    elif key == '馬場状態':
        return state2num(data)
    elif key == '性':
        return sex2num(data)
    elif key == '着順':
        return rank2label(data)
    else:
        try:
            return float(data) 
        except:
            return 0


def main():

    data_path = './data/keiba.csv'

    data_dict = pd.read_csv(data_path)

    keys = ['年', '競馬場コード', '開催回数', '日数', 'レース数', 'コース距離', 'コースタイプ', '天気', '馬場状態', '頭数',	'着順',	'枠', '馬番', '性', '年齢', '斤量', '人気', '単勝オッズ', '馬体重', '増減']


    new_columns = ['year', 'code', 'num', 'day', 'race', 'distance', 'type', 'weather', 'state', 'hourses', 'answer', 'waku', 'no', 'sex', 'age', 'weight', 'popularity', 'odds', 'hweight', 'updown']
    print(len(keys))
    print(len(new_columns))
    assert len(keys) == len(new_columns)

    new_list = []
    for i in tqdm(range(len(data_dict))):
        dat = []
        for k in keys:
            dat.append(str2num(k, data_dict[k][i]))
        new_list.append(dat)
        
    df = pd.DataFrame(new_list, columns=new_columns)

    df_for_train = df.query('year < 2019')
    df_for_valid = df.query('year == 2019')
    df_for_test  = df.query('year == 2020')

    df_for_train.to_csv('./data/train.csv')
    df_for_valid.to_csv('./data/valid.csv')
    df_for_test.to_csv('./data/test.csv')

if __name__ == '__main__':
    main()
