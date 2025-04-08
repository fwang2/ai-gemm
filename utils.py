
def get_available_gpu_device(threshold=50):
    """
    return a GPU device with less than 50% utilization
    and None if all are busy
    """
    
    try:
        output = subprocess.check_output(['nvidia-smi', 
                                          '--query-gpu=index,utilization.gpu', 
                                          '--format=csv,noheader,nounits'])
        gpu_utilizations = output.decode('utf-8').strip().split('\n')
        for gpu_info in gpu_utilizations:
            gpu_index, gpu_util = gpu_info.split(',')
            gpu_index = int(gpu_index.strip())
            gpu_util = int(gpu_util.strip())
            if gpu_util < threshold:
                return torch.device(f'cuda:{gpu_index}')
        warnings.warn("All GPUs have utilization above the threshold.")
        return None
    except subprocess.CalledProcessError as e:
        warnings.warn(f"Failed to run nvidia-smi: {e}")
        return None
    