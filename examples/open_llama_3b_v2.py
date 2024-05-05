import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Espresso import GPUAllocator

if __name__ == "__main__":
    slo = 1000
    modelname = "llama_3b"
    model = load_model_from_config(f'./model_config/{modelname}.json')
    gpus = load_gpus_from_config()
    start = time.time()
    min_cost = GPUAllocator.gpu_allocator(model,gpus,slo)
    end = time.time()
    print(f"min_cost = {min_cost} duration = {end - start}")