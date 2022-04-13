# Normalized Directional Motion Similarity

Implementation of Normalized Directional Motion Similarity (NDMS) introduced in
```
@inproceedings{tanke2021intention,
  title={Intention-based Long-Term Human Motion Anticipation},
  author={Tanke, Julian and Zaveri, Chintan and Gall, Juergen},
  booktitle={International Conference on 3D Vision},
  year={2021},
}
```
If this library is useful to you please cite `Intention-based Long-Term Human Motion Anticipation`.

## Install
```
pip install git+https://github.com/jutanke/ndms.git
```

## Usage

```python
from ndms.database import Data, Database


class SampleData(Data):
    """
    Storage of the data sequences that populate the evaluation Database.
    The sequences may have different lengths but must all have the same
    data dimenion.
    ! IMPORTANT ! Atm we only support 3D data, meaning that the dimension
    must be divisible by 3, meaning that each sequence should be 
      [n_frames x dim * 3]
    """
    def __init__(self, data):
        """
        :param data: list of sequences of variable length
        """
        self.data = data
        
    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)


test_sequences = SampleData(data=...)

kernel_size = 3  # nbr of frames for a motion word
db = Database(data=data, kernel_size=kernel_size)

framewise_ndms, _ = db.rolling_query(seq)


# you can define a transform function that is called for each motion word
# to for example normalize the data:
def transform(motion_word):
   """
   :param {kernel_size x dim*3}
   """
   return motion_word

db = Database(data=data, kernel_size=kernel_size, transform_data_fn=transform)


```
