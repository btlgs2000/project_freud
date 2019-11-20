from torch.utils.data import Dataset
import torch
from dataset import POSITIVE_RECORD, NEGATIVE_RECORD

class TorchTupleDataset(Dataset):
    def __init__(self, rows):
        ''' Un Dataset PyTorch di Tuple
        
        Attrs
        -----
        rows (Sequence[Tuple[Tuple[Sequence[float]], int]]): ogni tupla ha una label (1 se positivo, -1 se negativo)
        e una tupla di sequenze. Le sequenze possono anche essere array numpy.
        L'ultima è il record da classificare, le altre i record di esempio 
        '''
        super().__init__()
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        tupla, label = self.rows[index]
        tensor_tupla = tuple(torch.as_tensor(seq, dtype=torch.float) for seq in tupla)
        transformed_label = 1 if label == POSITIVE_RECORD else 0
        return torch.stack(tensor_tupla), torch.as_tensor(transformed_label, dtype=torch.float)
