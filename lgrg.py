import time
from torch.utils.tensorboard import SummaryWriter
import torch 
import torchtext
from torchtext import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir="logreg")

TEXT = data.Field()
LABEL = data.Field(sequential=False, unk_token=None)

train, val, test = torchtext.datasets.SST.splits(
    TEXT, LABEL,
    filter_pred=lambda ex: ex.label != 'neutral')

TEXT.build_vocab(train)
LABEL.build_vocab(train)

train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=10, device=device)

VOCAB_SIZE = len(TEXT.vocab)
CLASSES = len(LABEL.vocab) 

class BoWClassifier(nn.Module):  
    def __init__(self, num_labels, vocab_size):
        super(BoWClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        probs = F.log_softmax(self.linear(bow_vec), dim=1)
        #print(probs.shape)
        return probs

# make bag of words vectors for each batch
# init as torch tensor of 0s: [vocab_size, batch_len]
# forward: pass each column through linear layer
def make_bow_vector(batch_text):
    blen = batch_text.shape[1]
    seqlen = batch_text.shape[0]

    vec = torch.zeros(VOCAB_SIZE, blen)
    
    for b in range(blen):
        for s in range(seqlen):
            w = batch_text[s, b]           
            vec[w.item(), b] += 1

    vec = torch.transpose(vec, 0, 1) # [batch_len, vocab_size]
    #print(vec.shape)
    return vec

model = BoWClassifier(CLASSES, VOCAB_SIZE)

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(10):
    for batch in train_iter:
        model.zero_grad()
        bow_vector = make_bow_vector(batch.text)
        target = batch.label
        #print(target.shape)

        log_probs = model(bow_vector)
        #print(log_probs)
        #print(log_probs.shape)

        loss = loss_function(log_probs, target)
        #print(loss)
        #print(weight)
        writer.add_scalar('loss', loss, epoch)
        #print(epoch, loss)
        loss.backward()
        optimizer.step()
   
model.eval()
total_error = []
for batch in test_iter:
        bow_vector = make_bow_vector(batch.text)
        target = torch.LongTensor(batch.label)
        #print(bow_vector.shape)

        log_probs = model(bow_vector)
        #print(log_probs.shape)
        #print(log_probs)    
        #print(log_probs.max(1))
        _, argmax = log_probs.max(1)

        error = torch.abs(argmax - batch.label)
        error = sum(error)
        error = error.item()/len(batch.label) 
        #print(error)
        #writer.add_scalar('error', error, epoch)
        total_error.append(error)

total_error = sum(total_error)/len(total_error)
print('test error: ', total_error)
print("--- %s seconds ---" % (time.time() - start_time))
