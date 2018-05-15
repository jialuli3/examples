# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
import torch.multiprocessing as mp
#mp.set_start_method('spawn')

import data
import model
torch.backends.cudnn.enabled=False

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use(default:2)')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
#if torch.cuda.is_available():
#    if not args.cuda:
#        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
#torch.manual_seed(args.seed)
#device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################
corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
###############################################################################
# Try to train in multiple processes
###############################################################################

criterion = nn.CrossEntropyLoss()
hidden = None
total_loss = 0.

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden_eval = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden_eval = model(data, hidden_eval)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden_eval = repackage_hidden(hidden_eval)
    return total_loss / len(data_source)


def train(optimizer,min_batch,max_batch):
    # Turn on training mode which enables dropout.
    global total_loss,model
    model.train()
    #total_loss = 0.
    start_time = time.time()
    #ntokens = len(corpus.dictionary)
    #hidden = model.init_hidden(eval_batch_size)
    #for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
    for batch, i in enumerate(range(min_batch, max_batch, args.bptt)):
        def closure():
            global total_loss,hidden
            data, targets = get_batch(train_data, i)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = repackage_hidden(hidden)
            #print(hidden)
            model.zero_grad()
            optimizer.zero_grad()
            output, hidden = model(data, hidden)
            del data
            loss = criterion(output.view(-1, ntokens), targets)
            del output,targets
            loss.backward()
            total_loss += loss.item()
            return loss.item()
        optimizer.step(closure)

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        #total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                #elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
                elapsed * 1000 / args.log_interval, cur_loss, (cur_loss)))
            total_loss = 0
            start_time = time.time()


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
best_val_loss = None

#Construct optimizer
optimizer=optim.LBFGS(model.parameters(),lr=args.lr,history_size=5)

# At any point you can hit Ctrl + C to break out of training early.
def setUpTrain(min_batch,max_batch):
    global train_data,val_data,test_data,model,hidden
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    model=model.to(device)
    train_data=train_data.to(device)
    val_data=val_data.to(device)
    test_data=test_data.to(device)
    hidden = model.init_hidden(args.batch_size)
    pid=os.getpid()
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train(optimizer,min_batch,max_batch)
            val_loss = evaluate(val_data)
            print('-' * 89)
            print('{}| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(pid, epoch, (time.time() - epoch_start_time),
                    val_loss, (val_loss)))
            #                                   val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    if len(args.onnx_export) > 0:
        # Export the model in ONNX format.
        export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)


if __name__ == '__main__':
    model.share_memory()
    processes=[]
    sample_batch_size=int(train_data.size(0)/args.num_processes)
    for rank in range(args.num_processes):
        if rank<args.num_processes-1:
            p=mp.Process(target=setUpTrain,args=(rank*sample_batch_size,(rank+1)*(sample_batch_size)))
        else:
            p=mp.Process(target=setUpTrain,args=(rank*sample_batch_size,min((rank+1)*(sample_batch_size),train_data.size(0)-1)))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
