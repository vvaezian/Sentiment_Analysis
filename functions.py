import string 
import re 

def cleanup(text):
  '''Remove non-informative words and characters (punctuation, extra space, ...)'''
  tokens = text.split()
  filtered_tokens = ' '.join([ i for i in tokens if not i.startswith('http') ])

  # cleanup_chars = ''.join([ i for i in string.punctuation if i not in ("'", '/', '\\', '-', ':') ])  # puctuation chars minus "'" (it's used for example in "The Queen's Gambit")
  # remove_punctuation = str.maketrans(cleanup_chars, ' ' * len(cleanup_chars))
  # no_punctuation = filtered_tokens.translate(remove_punctuation)
  
  # no_extra_space = re.sub(' +', ' ', no_punctuation).strip()
  
  return filtered_tokens


def compress(data):
  
  def compress_string(string):

    if len(string) < 2:
      return string

    string_list = list(string) # reverse the string and 
    char_old = string_list[-1]
    for index, char in enumerate(string_list[-1::-1]):
      if char != char_old:
        break
      else:
        string_list[len(string_list) - index - 1] = None
    output = ''.join([i for i in string_list if i]) + char_old
    return output
  
  line_lists = [ i.split() for i in data ]
  compressed_tokens_list = [ [compress_string(i) for i in line_list ] for line_list in line_lists ]
  return [ ' '.join(i) for i in compressed_tokens_list ]

def map_pos(treebank_tag):
  from nltk.corpus import wordnet # pos
  if treebank_tag.startswith('J'):
    return wordnet.ADJ
  elif treebank_tag.startswith('V'):
    return wordnet.VERB
  elif treebank_tag.startswith('N'):
    return wordnet.NOUN
  elif treebank_tag.startswith('R'):
    return wordnet.ADV
  else:
    return wordnet.NOUN
  
def my_tokenize(line, print_emoji=False):  
  # Tokenizes a line of strings which may contain encoded emojis. In this case it converts them to strings 
  
  def my_decode(my_list):
    # Tries to decode the string as emoji and raises KeyError if not be able to

    import warnings
    warnings.filterwarnings('error')

    for i in my_list:
      if i[0] != 'x':
        raise KeyError
    my_str = ''.join('\\' + i for i in my_list)
    out = my_str.encode('latin').decode('unicode_escape').encode('latin').decode()
    return out


  def is_emoji_candidate(string):
    # Checks whether the string could be part of encoding of an emoji (e.g. x9f)

    hex_list = ['a', 'b', 'c', 'd', 'e', 'f', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    return string[0] == 'x' and string[1] in hex_list and string[2] in hex_list

  def preprocess(string):
    for i in ['b"','"',"\'","'",'\n', ':', ';', '`']:
      string = string.replace(i, '')
    for i in ['\\n', '.']:
      string = string.replace(i, ' ')
    return string

  def is_legal_token(string):
    # Final stage check, to decide what should be counted as a token after processing
    if string[0] == '@' or string[0] == '&' or 'http' in string:
      return False
    else:
      return True

  def my_split(string_containing_encoded_emoji):
    # Splits the string, seperating emoji encodings from words and components of emoji encodings themselves
    # E.g. transforms 'Hello \\xe2\\x9d\\xa4' to ['Hello', 'xe2', 'x9d', 'xa4']

    word_blocks = string_containing_encoded_emoji.replace('\\n', ' ').split('\\')
    tokens = []

    for item in word_blocks:
      if len(item) <= 3:
        tokens.append(item)
      elif is_emoji_candidate(item):
        tokens.append(item[:3])
        for i in item[3:].split():
          tokens.append(i)
      else:
        for i in item.split(' '):
          tokens.append(i)
    return tokens

  words_containing_split_decoded_emoji = my_split(preprocess(line))
  output = []
  cur_list = []
  
  for item in words_containing_split_decoded_emoji:
    if len(item) == 3 and is_emoji_candidate(item):
      cur_list.append(item)
      try:
        my_decode(cur_list)
        output.append(cur_list)
        cur_list = []
      except:
        if len(cur_list) > 4:
          cur_list = []
    else:
      cur_list = []
      output.append(item)
  
  tokens = []
  
  for item in output:
    if len(item) < 2:
      continue
    if type(item) is list:  # it is a list of components of an encoded emoji without '\', e.g. ['xf0','x9f','x92','x99']
      if print_emoji:
        tokens.append(''.join('\\' + i for i in item).encode('latin').decode('unicode_escape').encode('latin').decode())
      else:
        tokens.append(''.join(item))
    else:
      if is_legal_token(item):
        tokens.append(item.lower())
  return tokens

def convert_animated_emojis(data):
  converted_tokens_list = [my_tokenize(i) for i in data]
  converted_data = [ ' '.join(i) for i in converted_tokens_list ]
  return converted_data

def convert_text_emoji(data):

  emoji_list1 = [':)', ':-)', ':p', ':P', ':-p', ':-P', ':b', ':D', ":')", ":'-)", ':3', '<3', ':]', '=)', 'xD', 'XD', '8D', '=D', ':*', ':-*', ':x', ';)', ';-)', '^_^']
  emoji_list2 = [':(', ':-(', ":'(",':O', ':o', ':c', ':C', "D-':",'D:<','D:','D8','D;','D=','DX', ':\\', '=/', '=\\', ':X', ':$']
  emoji_list3 = [':|', ':-|']
  emoji_list4 = ['O_O','o-o','O_o','o_O','o_o','O-O']
  emoji_list_with_space = [' :/']
  
  output = []
  
  for line in data:
    for i in emoji_list1:
      line = line.replace(i, ' happy ')
    for i in emoji_list2 + emoji_list4 + emoji_list_with_space:
      line = line.replace(i, ' sad ')
    output.append(line)
  return output

def get_tokens(data):
  return ' '.join(data).split()

def show_emoji(string):
  '''expecting an emoji string without backslashes, e.g. "xf0x9fx98x82"'''
  full_string = ''
  if len(string) == 9:
    full_string = '\\' + string[:3] + '\\' + string[3:6] + '\\' + string[6:]
  elif len(string) == 12:
    full_string = '\\' + string[:3] + '\\' + string[3:6] + '\\' + string[6:9] + '\\' + string[9:]
  else:
    return string
  return full_string.encode('latin').decode('unicode_escape').encode('latin').decode()

def most_common(data, n, with_count=True):
  from collections import Counter
  if with_count:
    return [ i for i in Counter(get_tokens(data)).most_common(n) ] 
  else:
    return [ i[0] for i in Counter(get_tokens(data)).most_common(n) ] 

# test = 'b"\\xf0\\x9f\\x92\\x90\\xf0\\x9f\\x8c\\xb7@NiallOfficial\\xf0\\x9f\\x8c\\xb7\\xf0\\x9f\\x92\\x90\\n\\nHello Nialler! \\xf0\\x9f\\x92\\x95\\xf0\\x9f\\x92\\x95\\n\\nHow are you!?\\n\\nHope you\'re doing well! \\xf0\\x9f\\x98\\x81\\n\\nWill you please follow me!? You mean the \\xf0\\x9f\\x8c\\x8f to me. x37"\n'
# my_tokenize(test)
