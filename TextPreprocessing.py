from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import os
# Fetching the sub directories present in the root directory and setting the path to first subdirectory i.e,
# PHIlONTHROPISTS
os.chdir(r'C:\Users\mr.geek\Desktop\preprocessing')
path = os.getcwd()
dir_names = []
count = 0
for (dirpath, dirnames, filenames) in os.walk(path):
    dir_names.extend(dirnames)
path = path + "\\"
print(path)
paths = []
print(dir_names)
for directory in dir_names:
    paths.append(path + directory)
print(paths)
file_no = 0
for i in paths:
    for filename in os.listdir(i):
        lemma_list = []
        file_read = open(os.path.join(i, filename), 'r')
        read = file_read.read()

        # 1 - Following code removes the blank rows from the text
        text = os.linesep.join([s for s in read.splitlines() if s])

        # 2 - the following code changes the text to lower case
        text = text.lower()

        # 3- Word Tokenization
        tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
        tokens = tokenizer.tokenize(text)

        # 4- Stop words removal
        stw = stopwords.words('English')
        fileArr = [item for item in tokens if item not in stw]

        # 5 Word Net Lemmatizer requries Pos Tags to understand if the word is noun, verb or adjective
        tag_map = defaultdict(lambda: wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV
        final_word = []
        word_lemmatized = WordNetLemmatizer()
        file_no = file_no + 1
        for token, tag in pos_tag(fileArr):
            lemma_list.append(word_lemmatized.lemmatize(token, tag_map[0]))
        print("File NO {} with name {} is being written. Kindly Wait...".format(file_no, filename))
        file_write = open(os.path.join(i, filename), 'w')
        for lemma in lemma_list:
            file_write.write(lemma + " ")
    #     count = count + 1
    # #Changing the directory path
    # if count == 1:
    #     os.chdir(paths[count])
    #     path = os.getcwd()
    # elif count == 2:
    #     os.chdir(paths[count])
    #     path = os.getcwd()
    # elif count == 3:
    #     os.chdir(paths[count])
    #     path = os.getcwd()
    # elif count == 4:
    #     os.chdir(paths[count])
    #     path = os.getcwd()
