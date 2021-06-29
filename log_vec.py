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
writer = SummaryWriter(log_dir="logreg_vec")

TEXT = data.Field()
LABEL = data.Field(sequential=False, unk_token=None)

train, val, test = torchtext.datasets.SST.splits(
    TEXT, LABEL,
    filter_pred=lambda ex: ex.label != 'neutral')

TEXT.build_vocab(train, val, test, vectors='glove.6B.100d')
LABEL.build_vocab(train)

train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=10, device=device)

CLASSES = len(LABEL.vocab) 
EMB_SIZE = TEXT.vocab.vectors.shape[1]

class LinClassifier(nn.Module):  
    def __init__(self, num_labels, emb_size):
        super(LinClassifier, self).__init__()
        self.emb = nn.Embedding.from_pretrained(TEXT.vocab.vectors, freeze=True)
        self.linear = nn.Linear(emb_size, num_labels)

    def forward(self, batch_input):
        e = self.emb(batch_input) # [batch_size, seqlen, emb_size]
        e = e.sum(1)/len(batch_input[0]) # [batch_size, emb_size]  
        #e = F.relu(e)

        probs = F.log_softmax(self.linear(e), dim=1) # [batch_size, num_labels]
        #print(probs.shape)
        return probs

model = LinClassifier(CLASSES, EMB_SIZE)

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1):
    for batch in train_iter:
        model.zero_grad()
        batch_input_mat = torch.transpose(torch.LongTensor(batch.text), 0, 1)
        target = batch.label
        #print('target.shape', target.shape)

        log_probs = model(batch_input_mat)
        #print('log_probs', log_probs)
        #print('s', log_probs.shape)

        loss = loss_function(log_probs, target)    
        writer.add_scalar('loss', loss, epoch)
        #print(epoch, loss)        
        loss.backward()
        optimizer.step()

model.eval()
total_error = []
for batch in test_iter:
        bow_vector = torch.transpose(torch.LongTensor(batch.text), 0, 1)
        target = torch.LongTensor(batch.label)

        log_probs = model(bow_vector)
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
