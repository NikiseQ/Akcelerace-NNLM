# coding: utf-8
import argparse
import sys
import time
import math
import matplotlib.pyplot as plt
import plotly.plotly as py
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable, profiler

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
	#select seq_len words from position i
	data = Variable(source[i:i+args.seq_len], volatile=evaluation)
	return data

def evaluate(data_source):
	# Turn on evaluation mode which disables dropout.
	model.eval()
	start_time = time.time()
	times = []
	for batch, i in enumerate(range(0, data_source.size(0) - args.seq_len - 2, args.step)):
		with profiler.profile() as prof:
			data = get_batch(data_source, i, evaluation=True)
			output = model(data)
			if batch % args.print_batches == 0:
				elapsed = time.time() - start_time
				elapsed = elapsed * 1000 / args.print_batches

				times.append(elapsed)

				print('| {:5d}/{:5d} batches | ms/batch {:5.2f}'.format(
					batch + 1, len(data_source) // 1,
					elapsed))
				start_time = time.time()
		print(prof)
	return times

#----------------------------------------------------------------------------------------
if __name__ == "__main__":
	torch.manual_seed(1111)

	parser = argparse.ArgumentParser(description='PyTorch NN Language Model')
	parser.add_argument('--lr', type=float, default=0.01,
						help='initial learning rate')
	parser.add_argument('--initrange', type=float, default=0.1,
						help='range for weight initialization')
	parser.add_argument('--seq_len', type=int, default=2,
						help='sequence length')
	parser.add_argument('--print_batches', type=int, default=100,
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
	#model = model.NNModel(ntokens, args.size, args.size, args.seq_len, args.initrange, args.batch_size)
	
	#prof = torch.autograd.profiler.profile()
	

	NLLloss = nn.NLLLoss()
	lr = args.lr # learning rate

	# divide data into batch_size columns
	test_data = batchify(corpus.test, args.batch_size)
	
	print('*' * 80)

	print(args)

	#load model
	if args.load == '':
		print('error - missing model for loading')
		sys.exit();
		
	with open(args.load, 'rb') as f:
		model = torch.load(f)

	print('*' * 80)

	# At any point you can hit Ctrl + C to break out early.
	try:
		start_time = time.time()

		times = evaluate(test_data)

		print('-' * 80)
		print('| min: {:5.2f}ms | max: {:5.2f}ms | med: {:5.2f}ms | avg: {:5.2f}ms |'.format(
			min(times), max(times),
			numpy.median(times),
			sum(times)/float(len(times))))
		print('-' * 80)
		
		plt.hist(times, numpy.linspace(min(times), 15.00, 500), facecolor='green', alpha=0.9)
		plt.title("Times")
		plt.xlabel("Value")
		plt.ylabel("Frequency")
		plt.show()

	except KeyboardInterrupt:
		print('-' * 80)
		print('Exiting early')
