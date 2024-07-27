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
db = Database(data=test_sequences, kernel_size=kernel_size)

framewise_ndms, _ = db.rolling_query(seq)


# you can define a transform function that is called for each motion word
# to for example normalize the data:
def smpl_transform(motion_word):
   """
   :param {kernel_size x dim*3}
   """
   return motion_word


db = Database(data=data, kernel_size=kernel_size, transform_data_fn=transform)

```

NDMS can be used for SMPL-skeletons as follows:
```python
import numpy as np
import numpy.linalg as la
from ndms.database import Data, Database
from einops import rearrange


def apply_rotation_to_seq(seq, R):
    """
    :param seq: {n_frames x 18 x 3}
    :param R: {3x3}
    """
    R = np.expand_dims(R, axis=0)
    return np.ascontiguousarray(seq @ R)


def apply_normalization_to_seq(seq, mu, R):
    """
    :param seq: {n_frames x J x 3}
    :param mu: {3}
    :param R: {3x3}
    """
    is_flat = False
    if len(seq.shape) == 2:
        is_flat = True
        seq = rearrange(seq, "t (j d) -> t j d", d=3)

    if len(seq.shape) != 3 or seq.shape[2] != 3:
        raise ValueError(f"weird shape: {seq.shape}")

    mu = np.expand_dims(np.expand_dims(np.squeeze(mu), axis=0), axis=0)
    seq = seq - mu
    seq_rot = apply_rotation_to_seq(seq, R)

    if is_flat:
        seq_rot = rearrange(seq, "t j d -> t (j d)")
    return seq_rot


def get_normalization(left3d, right3d):
    """
    Get rotation + translation to center and face along the x-axis
    """
    mu = (left3d + right3d) / 2
    mu[2] = 0
    left2d = left3d[:2]
    right2d = right3d[:2]
    y = right2d - left2d
    y = y / (la.norm(y) + 0.00000001)
    angle = np.arctan2(y[1], y[0])
    R = rot.rot3d(0, 0, angle)
    return mu, R


class SMPLSkeletonData(Data):
    """
    Storage of the data sequences that populate the evaluation Database.
    The sequences may have different lengths but must all have the same
    data dimenion.
    ! IMPORTANT ! Atm we only support 3D data, meaning that the dimension
    must be divisible by 3, meaning that each sequence should be 
      [n_frames x 24 * 3]
    """
    def __init__(self, data):
        """
        :param data: list of sequences of variable length
        """
        self.data = data

    def n_dim(self):
      return 16 * 3
        
    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# you can define a transform function that is called for each motion word
# to for example normalize the data:
def transform(motion_word):
   """
   :param {kernel_size x dim*3}
   """
   jids_for_ndms = np.array([0, 1, 2, 4, 5, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21, 15], dtype=np.int64)
   motion_word = rearrange(motion_word, "t (j d) -> t j d", d=3)
   assert motion_word.shape[1] == 45 or motion_word.shape[1] == 24
   motion_word = np.copy(motion_word[:, :, [0, 2, 1]])  # we want the gravitational axis to be z, not y!
   left = motion_word[0, 1]  # left hip joint
   right = motion_word[0, 2]  # right hip joint
   mu, R = get_normalization(left, right)
   motion_word = apply_normalization_to_seq(motion_word, mu, R)
   return np.ascontiguousarray(
    rearrange(motion_word[:, jids_for_ndms], "t j d -> t (j d)")
   )
   return motion_word

# the data should be a list of SMPl skeletons [(n_frames x 45/24 x 3)] where the
# floor plane normal is the y-axis - if this is not the case you will have to
# transform your sequences such that the y-axis is the floor normal! This is important
# for porperly normalize the motion words during inference!
# The different motion sequences do not have to be of equal length!
test_sequences = SMPLSkeletonData(data=...)

# we suggest the kernel size to represent ~1/3second
kernel_size = 8  # nbr of frames for a motion word
db = Database(data=test_sequences, kernel_size=kernel_size, transform_data_fn=transform)

# seq is [n_frames x 24/45 x 3] - n_frames can be any length!
framewise_ndms, _ = db.rolling_query(seq)
```

