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
		
		self.hist_size = 0
		self.hist = []
		self.hid_val = []
		self.hist_index = 0
		self.hit_count = 0
		self.eval_count = 0
		

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
	
		if self.training == False: #eval
			#self.eval_count += 1
			for i, hist in enumerate(self.hist):
				if torch.equal(hist, input):
					#self.hit_count += 1
					#print(self.hit_count)
					#print(self.eval_count)
					print('hit')
					print('hit')
					print('hit')
					print('hit')
					print('hit')
					output = self.hid_val[i]

					output = self.decoder(output) # [32][|V|]
					output = self.Lsoftmax(output)
					return output # logsoftmax pre pravdepodobnosti
			
			emb = self.encoder(input)# emb - [2][32][150] 150 features pre 2 slov, input - 2 x index

			emb = torch.cat(torch.split(emb, 1), 2) # [1][32][300]
			emb = emb.view(self.batch_size, -1) # [32][300]

			output = self.emb2hid(emb)
			output = self.actF(output)
			
			if(self.hist_size != 300):
				self.hist.append(input)
				self.hid_val.append(output)
				self.hist_size += 1
			else:
				self.hist[self.hist_index] = input
				self.hid_val[self.hist_index] = output
				self.hist_index += 1
				if(self.hist_index == 300):
					self.hist_index = 0
				

			output = self.decoder(output) # [32][|V|]
			output = self.Lsoftmax(output)
			return output # logsoftmax pre pravdepodobnosti
