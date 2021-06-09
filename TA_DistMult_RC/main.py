import argparse
from dataset import Dataset
from trainer import Trainer
from tester import Tester
from params import Params

desc = 'Temporal KG Completion methods'
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('-dataset', help='Dataset', type=str, default='icews14', choices = ['icews14', 'icews05-15', 'gdelt'])
parser.add_argument('-model', help='Model', type=str, default='TA_DistMult_SED', choices = ['TA_DistMult_SED','TA_DistMult_MLP'])
parser.add_argument('-ne', help='Number of epochs', type=int, default=500, choices = [10,500,1000])
parser.add_argument('-bsize', help='Batch size', type=int, default=512, choices = [512])
parser.add_argument('-lr', help='Learning rate', type=float, default=0.001, choices = [0.001])
parser.add_argument('-reg_lambda', help='L2 regularization parameter', type=float, default=0.0, choices = [0.0])
parser.add_argument('-emb_dim', help='Embedding dimension', type=int, default=100, choices = [100,200])
parser.add_argument('-neg_ratio', help='Negative ratio', type=int, default=500, choices = [500])
parser.add_argument('-dropout', help='Dropout probability', type=float, default=0.4, choices = [0.0, 0.2, 0.4])
parser.add_argument('-save_each', help='Save model and validate each K epochs', type=int, default=100, choices = [10,100,200])
parser.add_argument('-se_prop', help='Static embedding proportion', type=float, default=1.0)

args = parser.parse_args()

dataset = Dataset(args.dataset)

params = Params(
    ne=args.ne, 
    bsize=args.bsize, 
    lr=args.lr, 
    reg_lambda=args.reg_lambda, 
    emb_dim=args.emb_dim, 
    neg_ratio=args.neg_ratio, 
    dropout=args.dropout, 
    save_each=args.save_each, 
    se_prop=args.se_prop
)

trainer = Trainer(dataset, params, args.model)
trainer.train()

# validating the trained models. we seect the model that has the best validation performance as the fina model
validation_idx = [str(int(args.save_each * (i + 1))) for i in range(args.ne // args.save_each)]
best_mrr = -1.0
best_index = '0'
model_prefix = "checkpoint/" + args.model + "1" + "/" + args.dataset + "/" + params.str_() + "_"

for idx in validation_idx:
    model_path = model_prefix + idx + ".chkpnt"
    tester = Tester(dataset, model_path, "valid")
    mrr = tester.test()
    if mrr > best_mrr:
        best_mrr = mrr
        best_index = idx

# testing the best chosen model on the test set
print("Best epoch: " + best_index)
model_path = model_prefix + best_index + ".chkpnt"

#model_path='checkpoint/DE_SimplE0/icews14/500_512_0.001_0.0_68_500_0.4_32_100_0.68_500.chkpnt'
print(model_path)
tester = Tester(dataset, model_path, "test")
tester.test()

