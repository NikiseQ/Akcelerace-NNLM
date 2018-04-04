import torch.nn as nn
import torch

class NNModel(nn.Module):

	# def __init__(self, ntoken, embed_size, nhid, seq_len, initrange):
	def __init__(self, ntoken, embed_size, nhid, seq_len, initrange, batch_size):
		super(NNModel, self).__init__()
		self.encoder = nn.Embedding(ntoken, embed_size) # ntoken - |V|, embed_size - pocet features
		
		self.emb2hid = nn.Linear(embed_size*seq_len, nhid)
		self.actF = nn.PReLU()

		self.decoder = nn.Linear(nhid, ntoken)

		self.initrange = initrange
		self.init_weights()
		
		self.Lsoftmax = nn.LogSoftmax(1)
		self.batch_size = batch_size

	def init_weights(self):
		self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
		self.decoder.bias.data.fill_(0)
		self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
		self.emb2hid.weight.data.uniform_(-self.initrange, self.initrange)

	def forward(self, input):
		emb = self.encoder(input)# emb - [2][32][150] 150 features pre 2 slov, input - 2 x index

		emb = torch.cat(torch.split(emb, 1), 2) # [1][32][300]
		emb = emb.view(self.batch_size, -1) # [32][300]

		output = self.emb2hid(emb)
		output = self.actF(output)

		output = self.decoder(output) # [32][|V|]
		output = self.Lsoftmax(output)
		return output # logsoftmax pre pravdepodobnosti
