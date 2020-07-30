import MeCab
#x86_64-linux-gnu以下に入っていたmecabをlib上に持っていったら通った。
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