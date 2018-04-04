# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model_tanh as model

#transforms data into bsz columns
def batchify(data, bsz):
	# Work out how cleanly we can divide the dataset into bsz parts
	nbatch = data.size(0) // bsz
	# Trim off any extra elements that wouldn't cleanly fit
	data = data.narrow(0, 0, nbatch * bsz)
	# Evenly divide the data across the bsz batches
	data = data.view(bsz, -1).t().contiguous()
	return data

#select data (seq_len long sentence) and target (word that follows data) from source
def get_batch(source, i, evaluation=False):
	#seq_len = min(args.seq_len, len(source) - 1 - i)
	#select seq_len words from position i
	data = Variable(source[i:i+args.seq_len], volatile=evaluation)
	#select next word as target
	target = Variable(source[i+args.seq_len])
	return data, target

def evaluate(data_source):
	# Turn on evaluation mode which disables dropout.
	model.eval()
	total_loss = 0

	nb_iters = 0
	for batch, i in enumerate(range(0, data_source.size(0) - 1, args.step)):
		if i + args.seq_len + 1 > data_source.size(0) - 1:
			break
		data, targets = get_batch(data_source, i, evaluation=True)
		output = model(data)
		loss = NLLloss(output, targets)
		total_loss += loss.data
		nb_iters += 1
	return total_loss[0] / nb_iters

def train(epoch):
	global lr

	# Turn on training mode which enables dropout.
	model.train()
	total_loss = 0
	#previous_loss = None
	start_time = time.time()

	for batch, i in enumerate(range(0, train_data.size(0) - 1, args.step)):
		if i + args.seq_len + 1 > train_data.size(0) - 1:
			break
		data, targets = get_batch(train_data, i)
		model.zero_grad() #optimizaer zero grad
		output = model(data) #output = log(P)
		
		loss = NLLloss(output, targets)

		loss.backward()

		#for p in model.parameters(): # sgd torch.optim
		#	p.data.add_(-lr, p.grad.data)
		optimizer.step()

		total_loss += loss.data

		if batch % args.print_batches == 0 and batch > 0:
			cur_loss = total_loss[0] / args.print_batches
			elapsed = time.time() - start_time
			print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:1.5f} | ms/batch {:5.2f} | '
					'loss {:5.2f} | ppl {:8.2f}'.format(
				epoch, batch, len(train_data) // 1, lr,
				elapsed * 1000 / args.print_batches, cur_loss, math.exp(cur_loss)))
			#if not previous_loss or previous_loss < cur_loss:
			#	lr /= 2.0
			#previous_loss = cur_loss
			total_loss = 0
			start_time = time.time()


if __name__ == "__main__":
	torch.manual_seed(1111)

	parser = argparse.ArgumentParser(description='PyTorch NN Language Model')
	parser.add_argument('--lr', type=float, default=0.1,
						help='initial learning rate')
	parser.add_argument('--initrange', type=float, default=0.1,
						help='range for weight initialization')
	parser.add_argument('--seq_len', type=int, default=2,
						help='sequence length')
	parser.add_argument('--print_batches', type=int, default=2000,
						help='print state after N batches')
	parser.add_argument('--epochs', type=int, default=6,
						help='number of epochs to do')
	parser.add_argument('--batch_size', type=int, default=32,
						help='batch size')
	parser.add_argument('--model_name', default='model',
						help='name of the model')
	parser.add_argument('--load', default='',
						help='load model to train')
	parser.add_argument('--step',type=int, default=1,
						help='size of step for moving through dataset')
	parser.add_argument('--size',type=int, default=150,
						help='size of model')
	args = parser.parse_args()

	corpus = data.Corpus('./data/penn')
	ntokens = len(corpus.dictionary) # ntoken - |V|

	# model = model.NNModel(|V|, embed_size, nhid, seq_len, initrange)
	model = model.NNModel(ntokens, args.size, args.size, args.seq_len, args.initrange, args.batch_size)

	NLLloss = nn.NLLLoss()
	lr = args.lr # learning rate

	#optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	optimizer = torch.optim.SGD(model.parameters(), lr=lr)

	# divide data into batch_size columns
	train_data = batchify(corpus.train, args.batch_size)
	val_data = batchify(corpus.valid, args.batch_size)
	test_data = batchify(corpus.test, args.batch_size)

	print('*' * 80)

	print(args)

	#load model
	if not args.load == '':
		print('opening model ' + args.load)
		with open(args.load, 'rb') as f:
			model = torch.load(f)

	print('*' * 80)

	best_val_loss = None

	# At any point you can hit Ctrl + C to break out of training early.
	try:
		for epoch in range(1, args.epochs + 1):
			epoch_start_time = time.time()

			train(epoch)
			#znova znova optimizer s novym LR
			val_loss = evaluate(val_data)

			print('-' * 80)
			print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
					'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
												val_loss, math.exp(val_loss)))
			print('-' * 80)

			# Save the model if the validation loss is the best we've seen so far.
			if not best_val_loss or val_loss < best_val_loss:
				with open(args.model_name, 'wb') as f:
					torch.save(model, f)
				best_val_loss = val_loss
			else:
				# Anneal the learning rate if no improvement has been seen in the validation dataset.
				lr /= 2.0
			#optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
			optimizer = torch.optim.SGD(model.parameters(), lr=lr) 

	except KeyboardInterrupt:
		print('-' * 80)
		print('Exiting from training early')

#------------Run on test data--------------
	test_loss = evaluate(test_data)

	print('=' * 80)
	print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
		test_loss, math.exp(test_loss)))
	print('=' * 80)