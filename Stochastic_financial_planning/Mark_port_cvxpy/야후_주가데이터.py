import os
import pandas as pd
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from datetime import datetime

currentPath = os.getcwd()
os.chdir('/Users/kimfr/OneDrive - unist.ac.kr/Optimization')
currentPath = os.getcwd()

 
start = datetime(2018,11,1)
end = datetime(2018,11,30)
 
index =  pd.read_excel('kospi200_index.xlsx')

df = web.DataReader('005930.KS', 'yahoo',start,end)

%matplotlib inline # 주피터 노트북 사용자는 이 명령어 추가해주세요
df['Close'].plot()