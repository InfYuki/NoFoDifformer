

from torch_geometric.datasets import Planetoid,WikiCS,Actor,WebKB


#dataset = Planetoid(root='', name='Cora')
#dataset = Planetoid(root='', name='citeseer')Pubmed
#dataset = Planetoid(root='', name='pubmed')
#dataset = WikiCS(root='./data/WikiCS')
#dataset = Actor(root='./data/Actor')
dataset = WebKB(root='./data/Texas', name='Texas')

print(dataset)


