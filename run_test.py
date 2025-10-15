import os

def run_test(params):
    p_name = params['probe_name'] if "/" not in params['probe_name'] else "_".join(params['probe_name'].split("/"))
    name = f"{params['detector_name']}_{p_name}_{params['threshold']}"
    log_dir = f"logs/test/{params['output_dir']}"
    results_dir = f"results/{params['output_dir']}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    log_name = f"{log_dir}/{name}"
    results_name = f"{results_dir}/{name}"

    datasets_str = ' '.join(params['test_datasets'])
    
    cmd = f"CUDA_VISIBLE_DEVICES={params['gpu_id']} nohup python3 -u test.py \
        --detector_name {params['detector_name']} \
        --probe_name {params['probe_name']} \
        --test_datasets {datasets_str} \
        --model_name {params['model_name']} \
        --format_id {params['format_id']} \
        --token_position {params['token_position']} \
        --layer_id {params['layer_id']} \
        --threshold {params['threshold']} \
        --output_name {results_name} \
        > {log_name}.log &"

    os.system(cmd)

# Base parameters configuration
test_params = {
    'detector_name': 'PIShield',
    'probe_name': 'data_llama3-8b_1_last/12',
    'test_datasets': [],
    'model_name': 'llama3-8b',
    'format_id': 1,
    'token_position': 'last',
    'layer_id': 12,
    'threshold': 0.5,
    'output_dir': 'main',
    'gpu_id': '3',
}

if __name__ == "__main__":
    datasets = []
    # get opi datasets
    for attack_pos in ['end']:
        for attack_strategy in ['naive', 'escape', 'ignore', 'fake_comp', 'combine', 'neural_exec', 'pleak', 'universal']:
            for target_task in ['sms_spam', 'sst2', 'mrpc', 'hsol', 'rte', 'jfleg','gigaword']:
                for inject_task in ['sms_spam', 'sst2', 'mrpc', 'hsol', 'rte', 'jfleg','gigaword']:
                    datasets.append(f"opi_malicious/{attack_pos}/{attack_strategy}/{target_task}_{inject_task}")
    for target_task in ['sms_spam', 'sst2', 'mrpc', 'hsol', 'rte', 'jfleg','gigaword']:
        datasets.append(f"opi_clean/{target_task}") 
    # get other datasets
    other_datasets={'dolly': ['dolly','dolly'],
                    'mmlu': ['mmlu','mmlu'],
                    'boolq': ['boolq','boolq'],
                    'hotelreview': ['hotelreview','close']
                    }
    for other_data in other_datasets:
        datasets.append(f'{other_data}_clean/{other_data}')
    for other_data, tasks_name in other_datasets.items():
        target_task, inject_task = tasks_name[0], tasks_name[1]
        for attack_pos in ['end']:
            for attack_strategy in ['naive', 'escape', 'ignore', 'fake_comp', 'combine', 'neural_exec', 'pleak', 'universal']:
                datasets.append(f"{other_data}_malicious/{attack_pos}/{attack_strategy}/{target_task}_{inject_task}")
    test_params['test_datasets'] = datasets

    test_params['probe_name']=f'data_{test_params["model_name"]}_{test_params["format_id"]}_last/{test_params["layer_id"]}'
    run_test(test_params)