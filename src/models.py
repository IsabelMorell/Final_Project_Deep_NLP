# deep learning libraries
import torch
from torch.nn.utils.rnn import pack_padded_sequence

class NERSA(torch.nn.Module):
    def __init__(
        self,
        embedding_weights: torch.Tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.5,
    ) -> None:
        """
        Constructor of the class NERSAModel, a LSTM model.

        Args:
            embedding_weights (torch.Tensor): Pre-trained word embeddings.
            hidden_size (int): hidden size of the layers.
            num_layers (int): The number of layers in the LSTM.
            dropout (float):  If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, 
            with dropout probability equal to dropout
        """
        super().__init__()

        # Determine the embedding dimension from the embedding weights
        embedding_dim: int = embedding_weights.shape[1]

        #  Create an embedding layer with the given pre-trained weights, use the Embedding.from_pretrained function
        self.embedding: torch.nn.Embedding = torch.nn.Embedding.from_pretrained(embedding_weights)

        # Initialize the LSTM layer
        self.lstm: torch.nn.LSTM = torch.nn.LSTM(embedding_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=True)

        # Create a linear layer for SA classification
        self.fc: torch.nn.Linear = torch.nn.Linear(in_features=2*hidden_size, out_features=3)  


    def forward(self, inputs: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        """
        This method returns a batch of logits. It is the output of the
        neural network.

        Args:
            inputs (torch.Tensor): The input tensor containing word indices.
                Dimensions: [batch, sequence_len]
            text_lengths (torch.Tensor): Tensor containing the lengths of texts in the batch.

        Returns:
            torch.Tensor: The output tensor after passing through the model.
        """

        # necesitamos que las frases tengan la misma longitud

        # Embed the input text using the embedding layer
        embedded: torch.Tensor = self.embedding(inputs)

        # TODO: Pack the embedded text for efficient processing in the LSTM
        packed_embedded: torch.Tensor = pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)


        # Initialize hidden state and memory cell
        self.h0: torch.Tensor = torch.empty(self.hidden_size)
        torch.nn.init.xavier_normal_(self.h0)  # torch.nn.init.xavier_uniform_(self.h0) si num_layer <= 2

        self.c0: torch.Tensor = torch.empty(self.hidden_size)
        torch.nn.init.xavier_normal_(self.c0)  # torch.nn.init.xavier_uniform_(self.c0) si num_layer <= 2

        # TODO: Pass the packed sequence through the LSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)


    