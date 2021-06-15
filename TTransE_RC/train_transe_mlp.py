import openke
from openke.config import Trainer, Tester
from openke.module.model import TTransE_MLP
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import numpy as np

# dataloader for training
                      #model change

dim=200
margin=8.0
save_path='./checkpoint/ttranse1.ckpt'
print(save_path)
    
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/icews14/",
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 1,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/icews14/", "link")

# define the model
transe = TTransE_MLP(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
    temp_tot = train_dataloader.get_temp_tot(),
	dim = dim,
	p_norm = 1, 
	norm_flag = True,
    )


# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = margin),
	batch_size = train_dataloader.get_batch_size()
)

#ipdb.set_trace()


# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 0.5, use_gpu = True)
trainer.run()
transe.save_checkpoint(save_path)


# test the model
transe.load_checkpoint(save_path)
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)
