#!/usr/bin/env python3

import argparse
import subprocess
import sys

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run evaluation for a specific dataset')
    parser.add_argument('--probe', type=str, default='data', help='Path to results files')
    parser.add_argument('--results_dir', type=str, default='main', help='Path to results files')
    parser.add_argument('--eval_type', type=str, default='all', help='Path to results files')
    parser.add_argument('--fpr_tasks', type=str, nargs='+', default=['opi', 'dolly', 'mmlu', 'boolq', 'hotelreview'], help='Path to results files')
    parser.add_argument('--fnr_tasks', type=str, nargs='+', default=['opi', 'dolly', 'mmlu', 'boolq', 'hotelreview'], help='Path to results files')
    args = parser.parse_args()

    # Base command and results file path
    base_cmd = "python3 -u evaluation.py"
    results_files = []
    if args.results_dir == 'main':
        detectors = [f'PIShield_{args.probe}_llama3-8b_1_last_12_0.5']
    for detector in detectors:
        results_files.append(f"results/{args.results_dir}/{detector}")

    commands = []
    if args.eval_type in ['fpr', 'all']:
        commands = [f"{base_cmd} --results_files {' '.join(results_files)} --func get_fpr_table --fpr_tasks {' '.join(args.fpr_tasks)}"]
    if args.eval_type in ['fnr', 'all']:
        for dataset in args.fnr_tasks:
            commands.append(f"{base_cmd} --results_files {' '.join(results_files)} --func get_fnr_table_per_dataset --test_data_name {dataset}")
        
    # Execute each command
    for cmd in commands:
        try:
            result = subprocess.run(cmd, shell=True, check=True, text=True)
            # print(f"Command completed successfully: {cmd}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()