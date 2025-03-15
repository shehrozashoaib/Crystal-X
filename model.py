import torch
import torch.nn as nn
from torch_geometric.nn import EdgeConv

class ConvLayer(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len, edge_hidden_dims=[256, 128], dropout=0.1):
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len

        # MLP to transform edge features (nbr_fea)
        self.edge_mlp = nn.Sequential(
            nn.Linear(nbr_fea_len, edge_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(edge_hidden_dims[0], edge_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(edge_hidden_dims[1], atom_fea_len),  # Output size matches node features
            nn.Dropout(dropout)
        )

        # EdgeConv layer to operate on node features
        self.edge_conv = EdgeConv(nn.Sequential(
            nn.Linear(64 * 2, 256),  # For concatenated source and target node features
            nn.ReLU(),
            nn.Linear(256, 64),  # Back to original feature size
        ),aggr='mean')
        
        # BatchNorm layers
        self.bn_input = nn.BatchNorm1d(atom_fea_len)  # BatchNorm for input node features
        self.bn_combined = nn.BatchNorm1d(atom_fea_len)  # BatchNorm for combined node and edge features
        self.bn_output = nn.BatchNorm1d(atom_fea_len)  # BatchNorm for final output features


    def forward(self, atom_fea, nbr_fea, edge_index):
        """
        Forward pass for EdgeConv with edge features.

        Parameters
        ----------
        atom_fea: torch.Tensor, shape (N, atom_fea_len)
            Node (atom) features.
        nbr_fea: torch.Tensor, shape (E, nbr_fea_len)
            Edge features (e.g., bond features).
        edge_index: torch.LongTensor, shape (2, E)
            Edge connectivity in COO format.

        Returns
        -------
        atom_out_fea: torch.Tensor, shape (N, atom_fea_len)
            Updated node (atom) features.
        """
        # Step 1: Transform edge features
        transformed_nbr_fea = self.edge_mlp(nbr_fea)  # Shape: (E, atom_fea_len)

        
        atom_fea = self.bn_input(atom_fea)  # Normalize node features
        
        
        # Step 2: Embed edge features into node features
        src, tgt = edge_index  # Source and target nodes
        #print(f"src.shape: {src.shape}")
        #print(f"tgt.shape: {tgt.shape}")
        
        transformed_nbr_fea = transformed_nbr_fea.view(-1, transformed_nbr_fea.size(-1))
        #print(f"transformed_nbr_fea.shape: {transformed_nbr_fea.shape}")
        # Create empty tensors for node features that will aggregate edge features
        edge_embedded_src = torch.zeros_like(atom_fea)  # Shape: (num_nodes, atom_fea_len)
        edge_embedded_tgt = torch.zeros_like(atom_fea)  # Shape: (num_nodes, atom_fea_len)

        # Aggregate edge features to both source and target nodes
        edge_embedded_src = edge_embedded_src.index_add_(0, src, transformed_nbr_fea)  # Source node aggregation
        edge_embedded_tgt = edge_embedded_tgt.index_add_(0, tgt, transformed_nbr_fea)  # Target node aggregation

        # Step 3: Combine node features with edge features (source + target)
        combined_fea = atom_fea + edge_embedded_src + edge_embedded_tgt  # Shape: (num_nodes, atom_fea_len)

        # Step 4: Apply EdgeConv on updated features
        atom_out_fea = self.edge_conv((combined_fea, atom_fea), edge_index)  # Ensure input is a PairTensor

        atom_out_fea = self.bn_output(atom_out_fea)
        
        atom_out_fea = atom_out_fea+combined_fea
        
        return atom_out_fea




class CrystalGraphConvNet(nn.Module):
    def __init__(self, orig_atom_fea_len, nbr_fea_len, atom_fea_len=64, n_conv=3, h_fea_len=256, n_h=2, dense_units=[128, 64]):
        """
        Initialize the Crystal Graph Convolutional Network.

        Parameters
        ----------
        orig_atom_fea_len: int
            Original atom feature length.
        nbr_fea_len: int
            Neighbor feature length.
        atom_fea_len: int, optional
            Atom feature length after embedding (default is 64).
        n_conv: int, optional
            Number of convolutional layers (default is 3).
        h_fea_len: int, optional
            Hidden feature length for fully connected layers (default is 128).
        n_h: int, optional
            Number of hidden layers (default is 2).
        dense_units: list, optional
            List of units for fully connected layers (default is [128, 64]).
        """
        super(CrystalGraphConvNet, self).__init__()

        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)

        # Create convolutional layers
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len) for _ in range(n_conv)])

        # Linear layer to transform atom features to hidden features
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)

        # Fully connected layers
        fc_layers = []
        input_size = h_fea_len
        for units in dense_units:
            fc_layers.append(nn.Linear(input_size, units))
            fc_layers.append(nn.ReLU())
            input_size = units
        self.fc_hidden = nn.Sequential(*fc_layers)

        # Output layer for final prediction
        self.fc_out = nn.Linear(dense_units[-1], 1)

    def create_edge_index(self, nbr_fea_idx):
        N, M = nbr_fea_idx.size()
        src = torch.arange(N, device=nbr_fea_idx.device).unsqueeze(1).expand(-1, M).reshape(-1)
        tgt = nbr_fea_idx.view(-1)
        return torch.stack([src, tgt], dim=0)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass of the Crystal Graph Convolutional Network.

        Parameters
        ----------
        atom_fea: torch.Tensor, shape (N, orig_atom_fea_len)
            Atom features (original).
        nbr_fea: torch.Tensor, shape (N, M, nbr_fea_len)
            Neighbor features (bonds).
        nbr_fea_idx: torch.LongTensor, shape (N, M)
            Indices of neighbors for each atom.
        crystal_atom_idx: torch.LongTensor
            Indexing information for pooling atoms in a crystal structure.

        Returns
        -------
        torch.Tensor
            Final output after fully connected layers.
        """
        # Embed atom features to the desired atom feature length
        atom_fea = self.embedding(atom_fea)

        # Generate edge_index for the graph connectivity
        edge_index = self.create_edge_index(nbr_fea_idx)

        # Apply convolutional layers
        for conv_layer in self.convs:
            atom_fea = conv_layer(atom_fea, nbr_fea, edge_index)

        # Pooling the atom features (mean over atoms in a crystal)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)

        # Pass through fully connected layers
        crys_fea = self.conv_to_fc(crys_fea)
        crys_fea = self.fc_hidden(crys_fea)

        # Output layer for prediction (e.g., formation energy or band gap)
        return self.fc_out(crys_fea)

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling operation to aggregate features from atoms in a crystal.
        Typically, this will sum or average the features of atoms in each crystal.

        Parameters
        ----------
        atom_fea: torch.Tensor, shape (N, atom_fea_len)
            Atom features after convolution.
        crystal_atom_idx: torch.LongTensor, shape (num_crystals, num_atoms_in_crystal)
            Indices of atoms in a crystal structure.

        Returns
        -------
        torch.Tensor, shape (num_crystals, atom_fea_len)
            Pooled features for each crystal.
        """
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True) for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)