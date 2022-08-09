import numpy as np
import math as m
from os.path import join, isfile
from annoy import AnnoyIndex
import numpy.linalg as la

from abc import ABC, abstractmethod

from ndms.ndms import ndms


class Data(ABC):
    @abstractmethod
    def __getitem__(self, index: int):
        """
        :return:
            np.ndarray
        """
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()


class Database:
    def __init__(
        self,
        data: Data,
        kernel_size: int,
        transform_data_fn=None,
        cache_fname: str = None,
    ):
        if cache_fname is not None and not cache_fname.endswith(".npz"):
            cache_fname = cache_fname + ".npz"

        self.kernel_size = kernel_size
        self.transform_data_fn = transform_data_fn

        if cache_fname is not None and isfile(cache_fname):
            obj = np.load(cache_fname)
            self.Meta = obj["Meta"]
            self.Seqs = obj["Seqs"]
            self.Orig_Seqs = obj["Orig_Seqs"]
        else:
            self.Meta = []
            self.Seqs = []
            self.Orig_Seqs = []
            for seqid in range(len(data)):
                seq = data[seqid]

                total_dim = seq.shape[1]
                assert int(m.ceil(total_dim / 3)) == int(m.floor(total_dim / 3))

                for frame in range((len(seq) - kernel_size) + 1):
                    sub_seq = seq[frame : frame + kernel_size].copy().astype("float32")
                    self.Orig_Seqs.append(sub_seq)
                    if transform_data_fn is not None:
                        sub_seq = transform_data_fn(sub_seq)
                    self.Seqs.append(sub_seq.flatten())
                    self.Meta.append((seqid, frame))

            if cache_fname is not None:
                np.savez(
                    cache_fname,
                    Seqs=self.Seqs,
                    Meta=self.Meta,
                    Orig_Seqs=self.Orig_Seqs,
                )

        n_dim = len(self.Seqs[0])
        self.lookup = AnnoyIndex(n_dim, "euclidean")
        for i, v in enumerate(self.Seqs):
            self.lookup.add_item(i, v)
        self.lookup.build(10)

    def __len__(self):
        return len(self.Seqs)

    def query(self, subseq):
        """
        :param subseq: {n_dim}
        """
        i, dist = self.lookup.get_nns_by_vector(
            vector=subseq, n=1, include_distances=True
        )
        i = i[0]
        dist = dist[0]
        true_seq = self.Seqs[i]
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
            sub_seq = subseq[frame : frame + kernel_size].copy().astype("float32")
            if self.transform_data_fn is not None:
                sub_seq = self.transform_data_fn(sub_seq)
            sub_seq = sub_seq.flatten()
            d, i = self.query(sub_seq)
            distances.append(d)
            identities.append(i)
        return distances, identities
