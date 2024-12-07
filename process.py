import re
p_d = './,][-")(~!#@^%$;*?&№∙^:<:>=_+\|`1°234}{567890'

int2name = {
    0: 'negative',
    1: 'neutral',
    2: 'positive'
}

def process_ner(response):
  unique_types = set()
  for j in range(len(response[0])):
    r1 = response[1][j]
    for i in range(len(r1)):
      unique_types.add(r1[i])

  dct = {item:[] for item in unique_types}

  for j in range(len(response[0])):
    temp = [response[0][j], response[1][j]]
    for i in range(len(temp[0])):
      if temp[1][i] != 'O':
        dct[temp[1][i]].append(temp[0][i])
  S = []
  for k in dct.keys():
    if k!='O':
      S.append(f"{k} : {', '.join(dct[k])}")
  return S

def preprocess(text):
    output = text.replace('\n', ' ').replace('\t', ' ').replace('\u200c', ' ')
    output = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', "", output)
    output = re.sub(r'^https?:\/\/.*[\r\n]*', '', output, flags=re.MULTILINE)
    for k in p_d:
        output = output.replace(k, ' ')
    output = output.replace('  ', ' ')
    return output.strip()
