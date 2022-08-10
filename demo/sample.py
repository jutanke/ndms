import sys

sys.path.insert(0, "./../")
sys.path.insert(0, "./../ndms")
sys.path.insert(0, "./")

import numpy as np
from ndms.database import Data, Database


class SampleData(Data):
    def __init__(self):
        super().__init__()
        # sample sequences with dim=3 but varying lengths
        seq1 = np.array(
            [
                [0.0, 0.1, 0.2],
                [0.1, 0.2, 0.3],
                [0.2, 0.3, 0.4],
                [0.3, 0.4, 0.5],
                [0.4, 0.5, 0.6],
            ]
        )
        seq2 = np.array(
            [
                [0.0, 0.2, 0.2],
                [0.1, 0.3, 0.3],
                [0.2, 0.4, 0.4],
                [0.3, 0.5, 0.5],
                [0.4, 0.6, 0.6],
                [0.5, 0.7, 0.7],
            ]
        )
        seq3 = np.array(
            [
                [0.0, 0.1, 0.1],
                [0.1, 0.2, 0.2],
                [0.2, 0.3, 0.3],
                [0.3, 0.4, 0.41],
                [0.4, 0.5, 0.51],
                [0.5, 0.6, 0.61],
                [0.6, 0.7, 0.71],
            ]
        )
        self.data = [seq1, seq2, seq3]

    def n_dim(self):
        return 3 * 2

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)


data = SampleData()

db = Database(data=data, kernel_size=2)


seq = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.4, 0.5, 0.6]])

print("#", len(db))

d, i = db.rolling_query(seq)

print("d", d)
print("i", i)

print(data.meta(kernel_size=2, db_index=i[0]))