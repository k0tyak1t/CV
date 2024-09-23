import torch
from tqdm.notebook import tqdm
import warnings

class Trainer:
	def __init__(self, model, loss_fn, optimizer=torch.optim.Adam, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), train_loader=None, test_loader=None):
		self.device = device
		self.model = model
		self.old_model = None
		self.params_ = model.parameters()
		self.loss_fn = loss_fn
		self.optimizer = optimizer(self.params_)
		self.train_losses = []
		self.test_losses = []
		self.train_loader = train_loader
		self.test_laoder = test_loader
		self.epoch_list = None
		self.apply_device()

		if self.train_loader is None or self.test_loader is None:
			warnings.warn('There is no data to train on!\nPlease, use `set_train_loader` and `set_test_loader` methods to set it or reassign class with params `train_loader` and `test_loader`.')

	def set_train_loader(self, train_loader):
		self.train_loader = train_loader
		self.apply_device()

	def set_test_loader(self, test_loader):
		self.test_loader = test_loader
		self.apply_device()

	def apply_device(self, device=None):
		self.device = device or self.device
		self.model = self.model.to(self.device)
		self.train_loader = self.train_loader.to(self.device)
		self.test_loader = self.test_loader.to(self.device)

	def set_optimizer(self, optimizer):
		self.optimizer = optimizer(self.params_)

	def set_model(self, model):
		warnings.warn("Your old model were saved. To reuse it use `backup_model` method.")
		self.model = model 
		self.apply_device()

	def backup_model(self):
		if self.old_model:
			self.model = self.old_model
		else:
			warnings.warn('There is no previous model!')

	def save_model(self, folder_name='model'):
		"""
		TODO: save model dict
		"""
		pass

	def load_model(self, file_name='model'):
		"""
		TODO: load model from dict
		"""

		self.model = ...

	def get_losses(self):
		return self.train_losses, self.test_losses

	def clear_losses(self):
		self.train_losses = []
		self.test_losses = []
		self.epoch_list = None

	def fit_epoch_(self):
		self.model.train()
		running_loss = 0
		n = 0

		for X, y in self.train_loader:
			y_pred = self.model(X)
			loss = self.loss_fn(y_pred, y)
			loss.backward()
			self.optimizer.step()
			self.optimizer.zero_grad()

			with torch.no_grad():
				running_loss += loss.item()
				n += 1

		return running_loss / n

	def eval_epoch_(self):
		self.model.eval()
		running_loss = 0
		n = 0

		for X, y in self.test_loader:
			with torch.no_grad():
				y_pred = model(X)
				loss = self.loss_fn(y_pred, y)
				running_loss += loss.item()
				n += 1

		return running_loss / n

	def fit(self, n_epoch=100, verbose=10):
		self.clear_losses()
		for epoch in tqdm(n_epoch):
			train_loss = self.fit_epoch_()
			test_loss = self.eval_epoch_()

			if epoch % verbose == (verbose - 1):
				print(f"{epoch+1} epoch | train loss: {train_loss:.6f} | test loss: {test_loss:.6f}")

			self.train_losses.append(train_loss)
			self.test_losses.append(test_loss)
			self.epoch_list = torch.arange(n_epoch)

	def learning_curve(self, get_ax=True):
		X = self.epoch_list or torch.arange(len(self.test_losses))

		fig, ax = plt.subplots(figsize=(10, 6))
		ax.plot(X, self.train_losses, label='train')
		ax.plot(X, self.test_losses, label='test')
		ax.legend()
		ax.grid()
		plt.show()

		if get_ax: return fig, ax