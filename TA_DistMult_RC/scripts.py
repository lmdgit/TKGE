import torch

def shredFacts(facts): #takes a batch of facts and shreds it into its columns
        
    heads      = torch.tensor(facts[:,0]).long().cuda()
    rels       = torch.tensor(facts[:,1]).long().cuda()
    tails      = torch.tensor(facts[:,2]).long().cuda()
    years1 = torch.tensor(facts[:,3]).long().cuda()
    years2 = torch.tensor(facts[:,4]).long().cuda()
    years3 = torch.tensor(facts[:,5]).long().cuda()
    years4 = torch.tensor(facts[:,6]).long().cuda()
    months = torch.tensor(facts[:,7]).long().cuda()
    days1 = torch.tensor(facts[:,8]).long().cuda()
    days2 = torch.tensor(facts[:,9]).long().cuda()
    return heads, rels, tails, years1, years2, years3, years4, months, days1, days2
