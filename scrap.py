import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import pandas as pd
import pickle

import os
import glob
import re
import argparse

class Scraping: 
    def __init__(self, 
        load_dir='./objects/',
        save_dir='./data/',
        start_year=2001,
        end_year=2020,
        ):

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.start_year = start_year
        self.end_year = end_year
        self.this_year = start_year

        self.field_name_list = ['札幌', '函館' , '福島', '新潟', '東京', '中山', '中京', '京都', '阪神', '小倉']

        self.dict = {}

    def _checkDirectory(self, p):
        if not os.path.isdir(p):
            os.makedirs(p)   
    
    def _checkFieldState(self):
        if self.dict["馬場状態"] == '良ダート' or self.dict["馬場状態"] == '稍重ダート' or self.dict["馬場状態"] == '重ダート' or self.dict["馬場状態"] == '不良ダート':
            return True
        return False
    
    def _setHorseInfoList(self, lst):
        self.this_list = lst
    
    def _setTable(self, tbl):
        self.this_table = tbl

    def _setFilenameInfo(self, filename):
        self.dict["ID"] = filename
        self.dict["年"] = filename[0:4]
        self.dict["競馬場コード"] = filename[4:6]
        self.dict["競馬場"] = self.field_name_list[int(self.dict["競馬場コード"]) - 1]
        self.dict["開催回数"] = filename[6:8]
        self.dict["日数"] = filename[8:10]
        self.dict["レース数"] = filename[10:]
    
    def _setRaceName(self, soup):
        self.dict["レース名"] = soup.select('.data_intro h1')[0].text.strip()
    
    def _setRaceInfo(self, soup):
       
        strings = soup.select('.data_intro p span')[0].text.replace('\xa0', '').replace(' ', '')
        data_list = re.split(r'[/:]', strings)

        self.dict["コース距離"] = re.sub("\\D", "", data_list[0])
        self.dict["コースタイプ"] = data_list[3]
        self.dict["天気"] = data_list[2]
        self.dict["馬場状態"] = data_list[4]
    
    def _setHorseInfo(self):
        
        self.dict["着順"] = self.this_list[0]
        self.dict["枠"] = self.this_list[1]
        self.dict["馬番"] = self.this_list[2]
        self.dict["馬名"] = self.this_list[3]
        self.dict["性"] = self.this_list[4][0]
        self.dict["年齢"] = self.this_list[4][1]
        self.dict["斤量"] = self.this_list[5]
        self.dict["騎手"] = self.this_list[6]
        self.dict["タイム"] = self.this_list[7]
        self.dict["着差"] = self.this_list[8].replace('/', '//')
        self.dict["上り"] = self.this_list[11]
        self.dict["人気"] = self.this_list[13]
        self.dict["単勝オッズ"] = self.this_list[12]
        self.dict["馬体重"] = self.this_list[14].split('(')[0]
        try:
            self.dict["増減"] = self.this_list[14].split('(')[1].replace(')', '')
        except:
            self.dict["増減"] = ''
    
    def _setTotalHourses(self, hourses_list):
        self.dict["頭数"] = len(hourses_list)

    def _initOddsInfo(self):
        odds_name_list = ["単勝", "複勝", "馬連", "ワイド", "馬単", "３連複", "３連単"]
        for n in odds_name_list:
            self.dict[n] = ''
            self.dict[n+'オッズ金額'] = ''
            
    def _setOddsInfo(self):
        self._initOddsInfo()

        odds = self._getOddsDataList()

        for o in odds:
            if o[0] == '単勝':
                self.dict["単勝"] = o[1]
                self.dict["単勝オッズ金額"] = o[2]
            elif o[0] == '複勝':
                self.dict["複勝"] = o[1].replace('-', '_')
                self.dict["複勝オッズ金額"] = o[2]
            elif o[0] == '馬連':
                self.dict["馬連"] = o[1].replace('-', '_')
                self.dict["馬連オッズ金額"] = o[2]
            elif o[0] == 'ワイド':
                self.dict["ワイド"] = o[1].replace('-', '_')
                self.dict["ワイドオッズ金額"] = o[2]
            elif o[0] == '枠連':
                self.dict["枠連"] = o[1].replace('-', '_')
                self.dict["枠連オッズ金額"] = o[2]
            elif o[0] == '馬単':
                self.dict["馬単"] = o[1].replace('-', '_')
                self.dict["馬単オッズ金額"] = o[2]
            elif o[0] == '３連複' or o[0] == '三連複':
                self.dict["３連複"] = o[1].replace('-', '_')
                self.dict["３連複オッズ金額"] = o[2]
            elif o[0] == '３連単' or o[0] == '三連単':
                self.dict["３連単"] = o[1].replace('-', '_')
                self.dict["３連単オッズ金額"] = o[2]
            else:
                print(o[0])

    def _getHorsesDataList(self):
        data = []
        try:
            rows = self.this_table[0].find_all('tr')
            for row in rows:
                dat = []
                for cell in row.find_all(['th', 'td']):
                    dat.append(cell.get_text().strip().replace('\n', '').replace(' ', ''))
                data.append(dat)
        except:
            pass

        return data

    
    def _getOddsDataList(self):
        data = []
        for i in range(1, 3):
            try:
                rows = self.this_table[i].find_all('tr')
                for row in rows:
                    dat = []
                    for cell in row.find_all(['th', 'td']):
                        for c in cell.select('br'):
                            c.replace_with('#')

                        dat.append(cell.get_text().strip().replace('\n', '').replace(' ', ''))
                    data.append(dat)
            except:
                pass

        return data


    def _loadPickle(self, path):
        with open(path, mode='rb') as f:
            data = pickle.load(f)
        return data

    def _saveCsv(self, dataframe):
        self._checkDirectory(self.save_dir)
        dataframe.to_csv(os.path.join(self.save_dir, 'keiba.csv'))


    def execute(self):

        data_mat = []
        
        for year in range(self.start_year, self.end_year + 1):
            print(year)
            file_list = glob.glob(os.path.join(self.load_dir, str(year), '*.pkl'))

            for pt in tqdm(file_list):
                filename = os.path.splitext(os.path.basename(pt))[0]

                self._setFilenameInfo(filename)

                data = self._loadPickle(pt)
                soup = BeautifulSoup(data.content, features='lxml')
               
                self._setRaceName(soup)
                self._setRaceInfo(soup)

                if self._checkFieldState():
                    continue

                self._setTable(soup.find_all('table'))

                self._setOddsInfo()

                hourses_list = self._getHorsesDataList()
                self._setTotalHourses(hourses_list)
                for h in hourses_list[1:]:
                    self._setHorseInfoList(h)
                    self._setHorseInfo()
                    temp = []
                    for key in self.dict.keys():
                        temp.append(self.dict[key])
                    
                    data_mat.append(temp)
        
        columns = list(self.dict.keys())
        df = pd.DataFrame(data_mat, columns=columns)

        self._saveCsv(df)
        

def args_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path', type=str, default='./objects/')
    parser.add_argument('--save_path', type=str, default='./data/')

    args = parser.parse_args()

    return args

def main(args):

    scp = Scraping(args.load_path, args.save_path)

    scp.execute()


if __name__ == '__main__':
    args = args_parser()

    main(args)

