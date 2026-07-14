import torch

class BaseEmbedder:
    def __init__(self, enable_cuda=True, weights='pretrained', checkpoint_path=None):
        self.device = torch.device("cuda:0" if enable_cuda and torch.cuda.is_available() else "cpu")
        self.model = None
        self.max_len = None
        self.sequence_type = None  # "rna" or "protein"
        self.weights = weights
        self.checkpoint_path = checkpoint_path

    def preprocess_sequence(self, sequence):
        """
        Override if needed.
        """
        return sequence.upper()

    def embed_window(self, seq_id, sequence):
        """
        Must return:
            np.array of shape (L, D)
        """
        raise NotImplementedError