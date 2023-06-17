import torch
from torch import nn
import pickle
import numpy as np


class EmotionRecognizerRNN(nn.Module):
  '''
  Recurrent Neural Network containing:
  - Embedding: turns words into vectors
  - Recurrent Component: RNN PyTorch component
  - Linear Layer: serving as an output layer
  - Softmax: output function for categorizing purposes
  '''
  def __init__(self, vocab_size: int, embedding: int, rnn_width: int) -> None:
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embedding)
    self.recurrent = nn.RNN(embedding, rnn_width, batch_first=True)
    self.linear = nn.Linear(rnn_width, 6)
    self.output_function = nn.Softmax(dim=1)
  
  def forward(self, text: torch.Tensor) -> torch.Tensor:
    embedded = self.embedding(text)
    _, hidden = self.recurrent(embedded)
    hidden = hidden.squeeze(0)
    output = self.linear(hidden)
    output = self.output_function(output)
    return output



class CustomTokenizer:
  def __init__(self, vocabulary: dict[str, int]) -> None:
    self.vocabulary = vocabulary
  
  def tokenize(self, text: str) -> torch.Tensor:
    text = text.lower()
    text = ''.join(c for c in text if c.isalpha() or c == ' ')
    words = text.split()
    ids = map(lambda e: self.vocabulary[e], words)
    ids = np.fromiter(ids, dtype=np.int32)
    return torch.from_numpy(ids).unsqueeze(0)


EMOTIONS = ['sadness', 'joy', 'fear', 'anger', 'love', 'surprise']


if __name__ == '__main__':
  tokenizer = pickle.load(open('./models/tokenizer.packed', 'rb'))
  model = EmotionRecognizerRNN(10000, 300, 128).to('mps')
  state = torch.load('./models/model_7755.pth')
  model.load_state_dict(state)
  model.eval()
  while True:
    prompt = input('Enter a sentence: \n')
    if prompt == 'exit': break
    feed = tokenizer.tokenize(prompt).to('mps')
    output = model(feed)
    emotion = EMOTIONS[output.argmax(dim=1).item()]
    print(f'Emotion: {emotion}')
