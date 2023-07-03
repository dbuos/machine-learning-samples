import torch
from language_utils import SimpleLanguageModel
from transformers import logging
from tqdm.auto import tqdm
import numpy as np

logging.set_verbosity_error()
if __name__ == '__main__':

    model_name = 'shakespeare04_checkpoint.pth'
    model_name2 = 'shakespeare_seq8_02_checkpoint.pth'
    weights = torch.load(model_name)['model_state_dict']
    model = SimpleLanguageModel(embeddings_from="bert-base-cased", rnn_size=128, rnn_layers=3)
    model.load_state_dict(weights)

    sequence = [138]
    n = 50
    beam_size = 3
    progress_bar = tqdm(range(n))
    device = torch.device('cuda:0')
    beams = [(np.log(1.0), sequence)]
    model.to(device)
    for i in range(n):
        mini_batch = []
        for beam in beams:
            mini_batch.append(beam[1])
        mini_batch = torch.tensor(mini_batch)
        mini_batch = mini_batch.to(device)
        y_predict = model(mini_batch)
        new_beams = []
        for j, logits in enumerate(y_predict):
            softmax = torch.nn.functional.softmax(logits, dim=0)
            beam = beams[j]
            for k in range(beam_size):
                next_word = torch.argmax(softmax).item()
                new_prob = beam[0] + np.log(softmax[next_word].item())
                new_seq = beam[1].copy()
                new_seq.append(next_word)
                new_beams.append((new_prob, new_seq))
                softmax[next_word] = 0
        new_beams.sort(key=lambda tup: tup[0], reverse=True)
        beams = new_beams[0:beam_size]
        progress_bar.update(1)
    print(beams[0])