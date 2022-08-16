import re
from collections import Counter, OrderedDict

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text.split()
    
def build_vocab(dataset):
    token_counts = Counter()
    for label, line in dataset:
        tokens = tokenizer(line)
        token_counts.update(tokens)
    return token_counts
