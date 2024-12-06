import subprocess
import torch

def sample_sdf(file, num_points, device='cpu'):
    command = ["/home/amuseth/dev/src/Neural-Sign-Distance-Learning/nsdf_live", file, str(num_points)]
    run_process = subprocess.run(command, capture_output=True, text=True)
    if run_process.returncode == 0:
        rows = run_process.stdout.strip().split("\n")
        data = [list(map(float, row.split())) for row in rows]
        tensor = torch.tensor(data).to(device)
        xyzs = tensor[:, :3]
        labels = tensor[:, -1].unsqueeze(-1)

        return xyzs, labels
    else:
        print("Failed to sample sdf on file: ", file)
        
sample_sdf('./Data/vdbs/airplane/train/airplane_0001.vdb', 10000)