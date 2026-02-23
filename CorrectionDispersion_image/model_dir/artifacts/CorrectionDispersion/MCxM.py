import torch
import torch.nn as nn

class MaskLayer(torch.nn.Module):
    def __init__(self, mask):
        super().__init__()
        mask= torch.tensor(mask, dtype=torch.float32)
        self.register_buffer('mask', mask)
    
    def forward(self, x):
        return x * self.mask

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
        self.verbose = True

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()
            if self.verbose:
                print(f"[EarlyStopping] Validation loss improved to {val_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement in validation loss for {self.counter} epochs")
            if self.counter >= self.patience:
                self.early_stop = True

class MCxM_CNN(torch.nn.Module):
    def __init__(self, mask, m=500, dropout_p=0.3, n_channel=2, n_global_features=6, n_mask_correction=3, wind_dim=2):
        super().__init__()

        # Parametri base
        self.m = m
        self.n_channel = n_channel
        self.n_global_features = n_global_features
        self.n_mask_correction = n_mask_correction
        self.wind_dim = wind_dim 
        self.mask_layer = MaskLayer(mask)
        self.dropout = nn.Dropout(p=dropout_p)

        # Calcolo dell'input size - flattened
        flat_local = self.n_channel * m * m            # mappe locali flattenate
        input_size = flat_local + self.wind_dim +  self.n_global_features
        hidden_size = 512

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),       # input_size → 512
            nn.BatchNorm1d(hidden_size),              # BN prima di ReLU aiuta a mantenere la media e la varianza dei dati stabili durante l’addestramento.
            nn.ReLU(),                                  #Metterla prima e dopo la ReLU serve a mantenere i dati ben distribuiti, riducendo problemi di saturazione o distribuzioni sbilanciate.
            #self.dropout,                          # Dropout per regolarizzazione
            #nn.BatchNorm1d(hidden_size),              # BN dopo ReLU

            nn.Linear(hidden_size, hidden_size),      # 512 → 512
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            #self.dropout,
            #nn.BatchNorm1d(hidden_size),

            nn.Linear(hidden_size, hidden_size),      # 512 → 512
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            #self.dropout,
            #nn.BatchNorm1d(hidden_size),

            nn.Linear(hidden_size, hidden_size),      # 512 → 512
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            #self.dropout,
            #nn.BatchNorm1d(hidden_size),
        )

        # DECODER: riportiamo a m²
        """self.decoder = nn.Sequential(
            nn.Linear(hidden_size, m*m),       # 512 → m*m
            nn.Sigmoid()  # normalizza output tra 0 e 1, utile per mappe binarie
        )"""
        self.decoder = nn.Linear(hidden_size, m*m)  # 512 → m*m

        # He initialization for weight
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)
        nn.init.kaiming_normal_(self.decoder.weight, nonlinearity='linear')
        nn.init.zeros_(self.decoder.bias)  

    def forward(self, gauss_disp, wind_features, global_features):
        
        B = gauss_disp.size(0) #batch size
        u = gauss_disp

        print(f"[Forward] Input gauss_disp shape: {gauss_disp.shape}")
        print(f"[Forward] Wind features shape: {wind_features.shape if wind_features is not None else None}")
        print(f"[Forward] Global features shape: {global_features.shape if global_features is not None else None}")

        for i in range(self.n_mask_correction):
            
            # --- MASKING LAYER
            # 1. Masking 
            u = self.mask_layer(u)  # shape: (B, m, m)
            print(f"[Forward][Step {i}] After mask_layer, u shape: {u.shape}")

            # 0. Concatenazione gauss_disp con variabili meteorogiche e globali
            x_flat = u.view(B, -1)  # (B, n_channel * m * m)
            print(f"[Forward][Step {i}] x_flat shape: {x_flat.shape}")

            um = torch.cat([x_flat, wind_features], dim=1)  # (B, n_channel*m*m + wind_dim)
            print(f"[Forward][Step {i}] After concatenating wind_features, um shape: {um.shape}")

            if global_features is not None:
                um = torch.cat([um,global_features], dim=1)  # shape: (B, m*m+2+n_global)
                print(f"[Forward][Step {i}] After concatenating global_features, um shape: {um.shape}")

            # --- CORRECTION NETWORK
            # 2. Flatten la mappa
            x = um.view(B, -1)  # shape: (B, m*m) -> (b, m^2)
            print(f"[Forward][Step {i}] Flattened for encoder, x shape: {x.shape}")

            # 3. Correzione tramite encoder-decoder
            x = self.encoder(x)
            print(f"[Forward][Step {i}] After encoder, x shape: {x.shape}")
            x = self.decoder(x) # shape: (B, m*m)
            print(f"[Forward][Step {i}] After decoder, x shape: {x.shape}")

            # 4. ricostruzione della mappa corretta
            u = x.view(B, self.m, self.m) # shape: (B, m, m)
            print(f"[Forward][Step {i}] Reconstructed map u shape: {u.shape}")

        return u
