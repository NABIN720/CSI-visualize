import torch
import pandas as pd
import os
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from laoder import CSIDataset
from attention import CNN_BiLSTM_Attention

def inference(data):
    print(f"Running inference on {len(data)} rows.")
    # data=pd.DataFrame(data)

    # print('data',data.shape)
    # print('data',data)
    # Model parameters
    input_dim = 64  
    hidden_dim = 512
    layer_dim = 2
    output_dim = 2
    dropout_rate = 0.3
    bidirectional = True
    SEQ_DIM = 1024
    DATA_STEP = 1
    BATCH_SIZE = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO)

    possible_paths = [
        "attention13M.pth",
        "python_utils/attention13M.pth",
        os.path.join(os.path.dirname(__file__), "attention13M.pth"),
        os.path.join(Path(__file__).parent.parent, "models", "attention13M.pth")
    ]

    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        print("Warning: Model file not found in any of these locations:")
        for path in possible_paths:
            print(f"- {os.path.abspath(path)}")
        print("\nContinuing without inference...")
        return None

    # Load model
    model = CNN_BiLSTM_Attention(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        layer_dim=layer_dim,
        dropout_rate=dropout_rate,
        bidirectional=bidirectional,
        output_dim=output_dim,
        seq_dim=SEQ_DIM
    ).to(device)

    # model.load_state_dict(torch.load("./python_utils/attention13.pth", map_location=device))

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).double()
    model.eval()


    val_dataset = CSIDataset(data, window_size=SEQ_DIM, step=DATA_STEP, is_training=False)
    val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    all_preds = []

    with torch.no_grad():  # No gradients needed for inference
        for x_batch in tqdm(val_dl, desc="Testing", total=len(val_dl)):
            if x_batch.size(0) != BATCH_SIZE:
                x_batch = x_batch[:BATCH_SIZE]  # Ensure batch consistency
            
            x_batch = x_batch.unsqueeze(1).double().to(device)
            out = model(x_batch)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())

    # Save Predictions
    predictions_df = pd.DataFrame(all_preds, columns=["Prediction"])
    predictions_df.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")