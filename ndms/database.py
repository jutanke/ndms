import numpy as np
import math as m
from os.path import join, isfile
from annoy import AnnoyIndex
import numpy.linalg as la

from abc import ABC, abstractmethod


from ndms.ndms import ndms


class Data(ABC):
    def __init__(self):
        self.meta_lookup = None  # db-index -> (seqid, frame)

    @abstractmethod
    def __getitem__(self, index: int):
        """
        :return:
            np.ndarray
        """
        raise NotImplementedError()

    @abstractmethod
    def n_dim(self):
        """
        return the final data dimension per frame
        AFTER potential transformations!
        """
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    def meta(self, kernel_size, db_index):
        """
        return index into sequences and frame
        """
        if self.meta_lookup is None:
            # build lookup
            self.meta_lookup = {}
            i = 0
            for seqid in range(len(self)):
                seq = self[seqid]
                n_frames = len(seq)
                for frame in range((n_frames - kernel_size) + 1):
                    self.meta_lookup[i] = (seqid, frame)
                    i += 1
        return self.meta_lookup[db_index]


class Database:
    @staticmethod
    def load_from_cache(
        cache_fname: str, kernel_size: int, dummy_dataset_fn, transform_data_fn=None
    ):
        """
        :param cache_fname:
        """
        assert isfile(cache_fname)
        return Database(
            data=dummy_dataset_fn(),
            kernel_size=kernel_size,
            transform_data_fn=transform_data_fn,
            cache_fname=cache_fname,
        )

    def __init__(
        self,
        data: Data,
        kernel_size: int,
        transform_data_fn=None,
        cache_fname: str = None,
    ):
        if cache_fname is not None and not cache_fname.endswith(".ann"):
            cache_fname = cache_fname + ".ann"

        self.kernel_size = kernel_size
        self.transform_data_fn = transform_data_fn
        n_dim = data.n_dim() * kernel_size
        self.lookup = AnnoyIndex(n_dim, "euclidean")
        if cache_fname is not None and isfile(cache_fname):
            self.lookup.load(cache_fname, prefault=False)
        else:
            i = 0
            for seqid in range(len(data)):
                seq = data[seqid]
                n_frames = len(seq)
                if len(seq.shape) > 2:
                    seq = seq.reshape((n_frames, -1))

                total_dim = seq.shape[1]
                assert int(m.ceil(total_dim / 3)) == int(m.floor(total_dim / 3))
                for frame in range((n_frames - kernel_size) + 1):
                    sub_seq = seq[frame : frame + kernel_size].copy()
                    if transform_data_fn is not None:
                        sub_seq = transform_data_fn(sub_seq)
                    self.lookup.add_item(i, sub_seq.flatten())
                    i += 1

            self.lookup.build(10)
            if cache_fname is not None:
                self.lookup.save(cache_fname)

    def __len__(self):
        return self.lookup.get_n_items()

    def query(self, subseq):
        """
        :param subseq: {n_dim}
        """
        i = self.lookup.get_nns_by_vector(vector=subseq, n=1, include_distances=False)
        i = i[0]

        true_seq = self.lookup.get_item_vector(i)

        kernel_size = self.kernel_size
        true_seq = np.reshape(true_seq, (kernel_size, -1))
        query_seq = np.reshape(subseq, (kernel_size, -1))
        dist = ndms(true_seq, query_seq, kernel_size=self.kernel_size)
        return dist, i

    def rolling_query(self, subseq):
        """
        :param subseq: {n_frames x dim}
        """
        kernel_size = self.kernel_size
        assert len(subseq) > kernel_size, str(subseq.shape)
        distances = []
        identities = []
        for frame in range((len(subseq) - kernel_size) + 1):
            sub_seq = subseq[frame : frame + kernel_size].copy()
            if self.transform_data_fn is not None:
                sub_seq = self.transform_data_fn(sub_seq)
            sub_seq = sub_seq.flatten()
            d, i = self.query(sub_seq)
            distances.append(d)
            identities.append(i)
        return distances, identities
