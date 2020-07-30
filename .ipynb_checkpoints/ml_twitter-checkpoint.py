"""
!pip install python-twitter

#教師データの生成
import twitter
import pandas as pd
import numpy as np
import os
os.chdir('パスを入力')

#TwitterのAPIキーを入力
api = twitter.Api(consumer_key='',
                  consumer_secret='',
                  access_token_key='',
                  access_token_secret='')

#検知しないツイート
#ツイートを日本語で新しいものから13500引っ張ってくる
count = 0
lst = []
for tweet in api.GetStreamSample(delimited=range(1,141), stall_warnings=True):
  if count == 0 :
    pass
    count += 1
  else:
    try:
      if tweet['lang'] == 'ja':
        lst.append(tweet['text'])
        if count == 13500 :
          break
        count += 1
    except :
      pass
  print(count)
general_df = pd.DataFrame({'content':lst})
general_df.to_csv('general_data.csv')

#検知するデータ
#都道府県の少年化がリプライしたツイートを取ってくる
def take_checked_tweet(user_name):
  kenkei = api.GetUserTimeline(screen_name=user_name,count=200)
  for tweet in kenkei:
    try:
      status_id = tweet.in_reply_to_status_id
      status = api.GetStatus(status_id=status_id)
      text = status.text
      tweet_dict[tweet.in_reply_to_screen_name] = text
    except:
      pass
  df = pd.DataFrame(list(tweet_dict.items()),columns=['user', 'content'])
  return df

#東京を除く46(都)道府県
hokkaido_kenkei = take_checked_tweet('HP_shonen')

aomori_kenieki = take_checked_tweet('AomoriPolice_SJ')
akita_kenkei = take_checked_tweet('AP_syoan')
iwate_kenkei = take_checked_tweet('iwate_syounen')
miyagi_kenkei = take_checked_tweet('miyagi_syounen')
fukushima_kenkei = take_checked_tweet('FP_syounen')
yamagata_kenkei = take_checked_tweet('ypymgt_shonen')

tochigi_kenkei = take_checked_tweet('TPJS_syounen')
yamanashi_kenkei = take_checked_tweet('YamanashiSyonen')
saitama_kenkei = take_checked_tweet('spp_syounen')
kanagawa_kenkei = take_checked_tweet('KPP_ikusei')
yamanashi_kenkei = take_checked_tweet('YamanashiSyonen')
ibraki_kenkei = take_checked_tweet('IBRK_syounen')
chiba_kenkei = take_checked_tweet('CP_syounen')
gunma_kenkei = take_checked_tweet('gunma_police_kj')

toyama_kenkei = take_checked_tweet('Toyama_syonen')
ishikawa_kenkei = take_checked_tweet('IP_syounen')
nigata_kenkei = take_checked_tweet('np_syounenka')
fukui_kenkei = take_checked_tweet('fukui_syounen')
gifu_kenkei = take_checked_tweet('GPsyounenka')
shizuoka_kenkei = take_checked_tweet('SPP_shonen')
nagano_kenkei = take_checked_tweet('NPP_syounen')
aichi_kenkei = take_checked_tweet('AP_syounen')
mie_kenkei = take_checked_tweet('Mpp_syounen')
kyoto_kenkei = take_checked_tweet('KPP_syounen')
osaka_kenkei = take_checked_tweet('osaka_syounen')
shiga_kenkei = take_checked_tweet('shigasyonen')
nara_kenkei = take_checked_tweet('nara_syounen')
wakayama_kenkei = take_checked_tweet('WPP_syounen')
hyogo_kenkei = take_checked_tweet('HPP_syounen')

okayama_kenkei = take_checked_tweet('OP_syonen')
yamaguchi_kenkei = take_checked_tweet('YP_syounen')
hiroshima_kenkei = take_checked_tweet('shoutai_hp')
tottori_kenkei = take_checked_tweet('shonen_tp')
shimane_kenkei = take_checked_tweet('shoutai_shimane')

ehime_kenkei = take_checked_tweet('EP_syounen')
kagawa_kenkei = take_checked_tweet('KppSyounen')
kochi_kenkei = take_checked_tweet('KP_shounen')
tokushima_kenkei = take_checked_tweet('TPP_syounen')

fukuoka_kenkei = take_checked_tweet('fukkei_syounen')
nagasaki_kenkei = take_checked_tweet('NSP_syounen')
saga_kenkei = take_checked_tweet('SKodomomamoru')
kumaoto_kenkei = take_checked_tweet('yuppi_syounen')
oita_kenkei = take_checked_tweet('opp_jinsyo')
kagoshima_kenkei = take_checked_tweet('kapo_saposen')
miyazaki_kenkei = take_checked_tweet('MP_Syounen')
okinawa_kenkei = take_checked_tweet('OPP_syounen')

df = pd.concat([hokkaido_kenkei, aomori_kenieki, akita_kenkei, iwate_kenkei, miyagi_kenkei, fukushima_kenkei,\
yamagata_kenkei, tochigi_kenkei, yamanashi_kenkei, saitama_kenkei, kanagawa_kenkei, yamanashi_kenkei, ibraki_kenkei, chiba_kenkei, gunma_kenkei,\
toyama_kenkei, ishikawa_kenkei, nigata_kenkei, fukui_kenkei, gifu_kenkei, shizuoka_kenkei, nagano_kenkei, mie_kenkei, aichi_kenkei\
kyoto_kenkei, osaka_kenkei, shiga_kenkei, nara_kenkei, wakayama_kenkei, hyogo_kenkei, \
okayama_kenkei, yamaguchi_kenkei, hiroshima_kenkei, shimane_kenkei, \
ehime_kenkei, kagawa_kenkei, kochi_kenkei, tokushima_kenkei, \
fukuoka_kenkei, nagasaki_kenkei, saga_kenkei, kumaoto_kenkei, miyazaki_kenkei, oita_kenkei, kagoshima_kenkei, \
okinawa_kenkei], axis=0)

df.to_csv('checked_data.csv')
"""


"""
#データを成形する
import pandas
import re

#@username、改行、URL、数字を削除
df = pd.read_csv('checked_data.csv',index_col=0)
df['content'] = df['content'].apply(lambda x : ''.join(x.split()[1:]) if str(x)[0] == '@' else x)
df['content'] = df['content'].replace("\n","",regex=True).replace("\u3000","",regex=True).replace("\t","",regex=True).replace(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)", "" ,regex=True)
df['content'] = df['content'].apply(lambda x : re.sub(r'[1-9]', '',x))general_df = pd.read_csv('general_data.csv',index_col=0,encoding="utf-8")

general_df = pd.read_csv('general_data.csv',index_col=0)
general_df['content'] = general_df['content'].apply(lambda x : ''.join(x.split()[1:]) if str(x)[0] == '@' else x)
general_df['content'] = general_df['content'].replace("\n","",regex=True).replace("\u3000","",regex=True).replace("\t","",regex=True).replace(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)", "" ,regex=True)
general_df['content'] = general_df['content'].apply(lambda x : ''.join(x.split()[2:]) if str(x)[0:2] == 'RT' else x)#RTusernameをけす
general_df['content'] = general_df['content'].apply(lambda x : re.sub(r'[1-9]', '',str(x)))#なぜこっちはstrでないといかんのかわかんない

#DataFrameにクラスを付ける
df['class'] = 1
general_df['class'] = 0

#userが不必要になったので除く
checked_df = df.iloc[:,1:3]

#検知データと不検知データをマージする
all_df = pd.concat([checked_df, general_df],axis=0)

#シャッフルし、コンテントに一文字もないツイートのを除く
shuff_all_df = all_df.sample(frac=1,random_state=42).reset_index(drop=True)
shuff_all_df = shuff_all_df[shuff_all_df['content'] != '']
"""

"""

#形態素分析
#googlecolabで行った場合のMeCabとipadic-neologd辞書のインストール方法
!apt-get -q -y install sudo file mecab libmecab-dev mecab-ipadic-utf8 git curl python-mecab > /dev/null
!git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git > /dev/null 
!echo yes | mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n > /dev/null 2>&1
!pip install mecab-python3 > /dev/null
!echo `mecab-config --dicdir`"/mecab-ipadic-neologd"
!sudo cp /etc/mecabrc /usr/local/etc/

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

X = pd.DataFrame(shuff_all_df['content'])
Y = pd.DataFrame(shuff_all_df['class'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=42)

#分かち書き
def tokenize(text):
    tagger = MeCab.Tagger("-Ochasen")
    node = tagger.parseToNode(text)
    result = []
    while node:
      features = node.feature.split(',')
      if features[0] != 'BOS/EOS':
        if (features[0] not in ['助詞', '助動詞', '記号']) and not (features [0] == '名詞' and features[1]=='サ変接続')\
        and not (features[1]== '数') and not (node.surface == "ｧｨ" or node.surface == "д"):
          token = features[6] if features[6] != '*' else node.surface
          if token.isalpha():
            token = token.lower()
            if token == 'p':
              token = 'パパ'
          result.append(token)
      node = node.next
    return result

#ベクトル化
vec_count = CountVectorizer(analyzer=tokenize,min_df=4)
vec_count.fit(X_train['content'])
X_train_vec = vec_count.transform(X_train['content'])
X_test_vec = vec_count.transform(X_test['content'])

#学習する
model = BernoulliNB()
model.fit(X_train_vec, Y_train['class'])

#モデルを保存する
import pickle
with open('model.pickle', mode='wb') as fp:
    pickle.dump(model, fp)

import pickle
with open('vectorizer.pickle', mode='wb') as fp:
    pickle.dump(vec_count, fp)
"""
from tokenizer import tokenize
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')
import sys

def main(arg):
    
    if len(arg) == 1:
        data = np.array(['控えめに言ってレポート多すぎる短いのだけど毎日1本書いてる気が...',#自分のツイート
                         #少年課が警告リプライをしたツイート(教師データとして使用していない新しいもの)
                         '今日大阪や京都や滋賀で会える女の子はいませんか？',
                         '高畑駅付近で会える人居ますか？',
                         '今日2から会える方いませんか？礼儀ないアカとドタキャン冷やかしアカ確認済みです！dmください！',
                         '愛知県できょう30日、新たに少なくとも146人の感染が確認されたことがわかった。このうち、名古屋市では108人の感染が確認されていて、名古屋市で感染者数が100人を超えるのは初めて。'])
    else:
        data = np.array(arg[1:])
        
    df_data = pd.DataFrame(data, columns=['text'])
    with open('./model/vectorizer.pickle', mode='rb') as fp:
        vec = pickle.load(fp)
    input_vec = vec.transform(df_data['text'])
    with open('./model/model.pickle', mode='rb') as fp:
        clf = pickle.load(fp)
    
    input_vec = vec.transform(df_data['text'])
    #危険 : 1, 危険ではない : 0
    return clf.predict(input_vec)

if __name__ == "__main__":
    print(main(arg=sys.argv))