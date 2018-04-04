import torch.nn as nn
import torch

class NNModel(nn.Module):

	# def __init__(self, ntoken, embed_size, nhid, seq_len, initrange):
	def __init__(self, ntoken, embed_size, nhid, seq_len, initrange, batch_size):
		super(NNModel, self).__init__()
		self.encoder = nn.Embedding(ntoken, embed_size) # ntoken - |V|, embed_size - pocet features
		
		self.emb2hid = nn.Linear(embed_size*seq_len, nhid)
		self.actF = nn.Tanh()
		#self.actF = nn.PReLU()

		self.decoder = nn.Linear(nhid, ntoken)

		self.initrange = initrange
		self.init_weights()
		
		self.Lsoftmax = nn.LogSoftmax(1)
		self.batch_size = batch_size

		self.projectionM1 = []
		self.projectionM2 = []
		self.ntoken = ntoken
		self.seq_len = seq_len
		self.nhid = nhid
		self.firtsEval = False

	def init_weights(self):
		self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
		self.decoder.bias.data.fill_(0)
		self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
		self.emb2hid.weight.data.uniform_(-self.initrange, self.initrange)

	def forward(self, input):
		if self.training == True: #train
			emb = self.encoder(input)# emb - [2][32][150] 150 features pre 2 slov, input - 2 x index

			emb = torch.cat(torch.split(emb, 1), 2) # [1][32][300]
			emb = emb.view(self.batch_size, -1) # [32][300]

			output = self.emb2hid(emb)
			output = self.actF(output)

			output = self.decoder(output) # [32][|V|]
			output = self.Lsoftmax(output)
			return output # logsoftmax pre pravdepodobnosti

		if not self.firtsEval:
			#print(self.ntoken)
			embeddings = torch.LongTensor(self.seq_len, self.ntoken).zero_()
			for j in range(0, self.seq_len):
				for v in range(0, self.ntoken):
					embeddings[j][v] = v
					#b = torch.autograd.variable.Variable(torch.zeros(150))
			#print(embeddings)
			#self.projectionM.append(self.encoder(torch.autograd.variable.Variable(a)))
			embeddings = self.encoder(torch.autograd.variable.Variable(embeddings))
			#print(embeddings)

			
			torch.t
			weights = torch.t(self.emb2hid.weight)
			#print(weights)
			#print(torch.t(weights))
			
			self.projectionM1 = torch.addmm(self.emb2hid.bias, embeddings[0], weights[0:150])
			self.projectionM2 = torch.mm(embeddings[1], weights[150:300])

			#self.projectionM = embeddings

			self.firtsEval = True

		M1 = self.projectionM1.index_select(0, input[0]).data
		M2 = self.projectionM2.index_select(0, input[1]).data

		output = torch.add(M1, 1, M2)
		output = torch.autograd.variable.Variable(output)

		output = self.actF(output)

		output = self.decoder(output) # [32][|V|]
		output = self.Lsoftmax(output)
		return output # logsoftmax pre pravdepodobnosti
