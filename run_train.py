import os

def run_train(params):
    os.makedirs('logs/train', exist_ok=True)
    log_name = f"logs/train/train_{params['data_name']}_{params['model_name']}_{params['format_id']}_{params['token_position']}"
    cmd = f"CUDA_VISIBLE_DEVICES={params['gpu_id']} nohup python3 -u train.py \
        --data_name {params['data_name']} \
        --layer_id {params['layer_id']} \
        --model_name {params['model_name']} \
        --format_id {params['format_id']} \
        --token_position {params['token_position']} \
        > {log_name}.log &"
    
    print(f"Running: {cmd}")
    os.system(cmd)

# Base parameters configuration
train_params = {
    'model_name': 'llama3-8b',
    'layer_id': 12,
    'format_id': 1,
    'token_position': 'last',
    'data_name': 'data',
    'gpu_id': '3'
}

if __name__ == "__main__":
    run_train(train_params)