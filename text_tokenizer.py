import re
from collections import Counter, OrderedDict

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text.split()
    
def build_vocab(dataset):
    token_counts = Counter()
    i = 0
    for label, lines in dataset:
        print(i)
        i += 1
        tokens = tokenizer(lines)
        token_counts.update(tokens)
        return token_counts

if __name__ == '__main__':
    text = '''<h1>Hola :) como estas ?</h1>'''
    print(tokenizer(text))