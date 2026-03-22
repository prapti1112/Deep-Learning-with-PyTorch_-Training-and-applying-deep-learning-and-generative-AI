# part1_core_pytorch/ch04_data_rep/02_volumetric_ct.py
import torch
import imageio

def main():
    # 4.2.1 Loading a specialized format (DICOM/MetaIO)
    # Note: Requires sample DICOM series
    dir_path = "data/p1ch4/volumetric-dicom/2-LUNG 3.0  B70f-04083"
    try:
        vol_arr = imageio.volread(dir_path, 'DICOM')
        print(f"Volume Shape (DHW): {vol_arr.shape}")
        
        # Convert to tensor and add channel dimension (C=1)
        vol = torch.from_numpy(vol_arr).float()
        vol = torch.unsqueeze(vol, 0)  # Shape: (1, D, H, W)
        print(f"Tensor Shape (CDHW): {vol.shape}")
    except Exception as e:
        print(f"Skipping volumetric load (data missing): {e}")

if __name__ == "__main__":
    main()