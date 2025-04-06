# deep learning libraries
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class NERSA(torch.nn.Module):
    def __init__(
        self,
        embedding_weights: torch.Tensor,
        hidden_size: int,
        num_layers: int,
        num_NER_labels: int, 
        num_SA_labels: int = 3,
        dropout: float = 0.5,
    ) -> None:
        """
        Constructor of the class NERSAModel, a LSTM model.

        Args:
            embedding_weights (torch.Tensor): Pre-trained word embeddings.
            hidden_size (int): hidden size of the layers.
            num_layers (int): The number of layers in the LSTM.
            num_NER_labels (int): number of posible NER labels.
            num_SA_labels (int): number of sentiments.
            dropout (float):  If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, 
            with dropout probability equal to dropout
        """
        super().__init__()

        # Determine the embedding dimension from the embedding weights
        embedding_dim: int = embedding_weights.shape[1]

        #  Create an embedding layer with the given pre-trained weights, use the Embedding.from_pretrained function
        self.embedding: torch.nn.Embedding = torch.nn.Embedding.from_pretrained(embedding_weights)

        # Initialize the LSTM layer
        self.lstm: torch.nn.LSTM = torch.nn.LSTM(embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)

        # Create a linear layer for NER
        self.fc_ner = torch.nn.Linear(2 * hidden_size, num_NER_labels)

        # Create a linear layer for SA classification
        self.fc_sa: torch.nn.Linear = torch.nn.Linear(in_features=2*hidden_size, out_features=num_SA_labels)  


    def forward(self, inputs: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        """
        This method returns a batch of logits. It is the output of the
        neural network.

        Args:
            inputs (torch.Tensor): The input tensor containing word indices.
                Dimensions: [batch, sequence_len]
            text_lengths (torch.Tensor): Tensor containing the lengths of texts in the batch.

        Returns:
            torch.Tensor: The output tensor after passing through the model with the logits of NER labels.
                Dimensions: [batch, sequence_len, num_NER_labels]
            torch.Tensor: logits of Sentiment Analysis (SA)
                Dimensions: [batch, num_SA_labels]
        """

        # Embed the input text using the embedding layer
        embedded: torch.Tensor = self.embedding(inputs)

        # Pack the embedded text for efficient processing in the LSTM
        packed_embedded: torch.Tensor = pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)

        batch_size: int = inputs.shape[0]

        # Initialize hidden state and memory cell
        h0: torch.Tensor = torch.empty((2*self.num_layers, batch_size, self.hidden_size), dtype=torch.float32, device=inputs.device)
        c0: torch.Tensor = torch.empty((2*self.num_layers, batch_size, self.hidden_size), dtype=torch.float32, device=inputs.device)
        
        torch.nn.init.xavier_normal_(h0)  # torch.nn.init.xavier_uniform_(self.h0) si num_layer <= 2
        torch.nn.init.xavier_normal_(c0)  # torch.nn.init.xavier_uniform_(self.c0) si num_layer <= 2

        # Pass the packed sequence through the LSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded, (h0, c0))

        # Unpack LSTM output
        output_unpacked, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Sentiment Analysis
        pooled_hidden_state: torch.Tensor = hidden.mean(dim=0)  # [-1, :, :]

        # si con solo una lineal no funciona entonces pondremos un MLP

        return self.fc_ner(output_unpacked), self.fc_sa(pooled_hidden_state)