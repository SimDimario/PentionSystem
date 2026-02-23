import torch
import numpy as np
from CorrectionDispersion.MCxM import MCxM_CNN
import os
import logging

logger = logging.getLogger("CorrectionDispersion")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def calculate_mean_direction(wind_dir_array):
    # Convert to radians
    wind_dir_rad = np.radians(wind_dir_array)

    # Calculate directional vectors
    cos_vals = np.cos(wind_dir_rad)
    sin_vals = np.sin(wind_dir_rad)

    # Calculate mean
    mean_cos = np.mean(cos_vals)
    mean_sin = np.mean(sin_vals)

    return mean_cos, mean_sin

def load_model(binary_map, m=500, device=None, pretrained_path=None):
    model_path = os.path.join(SCRIPT_DIR, "models", "mcxm_cnn_model.pth")
    logger.info(f"Loading MCxM_CNN model from {model_path} (device={device})")

    loaded_model = MCxM_CNN(binary_map, m=m, n_channel=1, n_mask_correction=3, wind_dim=2, n_global_features=0).to(device)

    try:
        loaded_model.load_state_dict(torch.load(model_path, map_location=device))
        loaded_model.eval()
        logger.info("Model loaded and set to eval mode.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e
    return loaded_model

def correct_dispersion(wind_dir, wind_speed, concentration_map, building_map, global_feature=None, device=None, m=500, pretrained_path=None):
    logger.info("Starting dispersion correction...")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.debug(f"Using device: {device}")

    logger.info(f"building map: {building_map.shape}")

    model = load_model(building_map, m=m, device=device, pretrained_path=pretrained_path)

    cm_agg=np.mean(concentration_map, axis=2) #shape: (500,500
    mc = torch.tensor(cm_agg, dtype=torch.float32, device=device).unsqueeze(0)  # (1, m, m)
    logger.debug(f"Concentration map tensor shape: {mc.shape}")

    #local_maps = building_map.unsqueeze(0)  # [1, 1, m, m]

    wind_dir_cos , wind_dir_sin = calculate_mean_direction(wind_dir)
    wind_dir_angle = np.arctan2(wind_dir_sin, wind_dir_cos)
    degree_angle = np.degrees(wind_dir_angle)

    logger.debug(f"Wind direction: {degree_angle}°, ")

    wind_features = torch.tensor([[wind_speed, degree_angle]], dtype=torch.float32, device=device)
    logger.debug(f"Wind features tensor: {wind_features}")

    if global_feature is not None:
        global_features = torch.tensor(global_feature, dtype=torch.float32, device=device).unsqueeze(0)
        logger.debug(f"Global features tensor shape: {global_features.shape}")
    else:
        global_features = None

    logger.info("Running model inference...")
    with torch.no_grad():
        try:
            output = model(mc, wind_features, global_features)
            output = output.detach().cpu().numpy()[0] # togli batch, shape (m, m)
            logger.info(f"Inference completed. Output shape: {output.shape}")
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            raise e

    return output