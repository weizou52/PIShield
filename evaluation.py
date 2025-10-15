from sklearn.metrics import accuracy_score
from utils import jload
import argparse 
import numpy as np

def get_evaluation_metrics(y_pred, y_test):
    assert len(y_test) == len(y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    if len(y_test) != len(y_pred):
        raise ValueError("The lengths of true labels and predicted labels must be the same.")
    
    for label in y_test + y_pred:
        if label not in {0, 1}:
            raise ValueError("Labels must be either 0 or 1.")
    
    TP, FP, TN, FN = 0, 0, 0, 0
    
    for pred, true in zip(y_pred, y_test):
        if pred == 1:
            if true == 1:
                TP += 1
            else:
                FP += 1
        else:
            if true == 1:
                FN += 1
            else:
                TN += 1
    
    # Calculate False Positive Rate (FPR)
    denominator_fpr = FP + TN
    fpr = 0.0 if denominator_fpr == 0 else FP / denominator_fpr
    
    # Calculate False Negative Rate (FNR)
    denominator_fnr = FN + TP
    fnr = 0.0 if denominator_fnr == 0 else FN / denominator_fnr
    
    return accuracy, fpr, fnr

def evaluate(results_file, test_data_name):
    if test_data_name is None:
        raise ValueError("Test data name is required")
    results = jload(results_file)
    labels=jload(f"TestData/{test_data_name}/label")
    y_pred = results[test_data_name]
    accuracy, fpr, fnr = get_evaluation_metrics(y_pred, labels)
    print(f"Task: {test_data_name}")
    print(f"Accuracy: {accuracy}")
    print(f"False Positive Rate (FPR): {fpr}")
    print(f"False Negative Rate (FNR): {fnr}")
    print("\n")
 
def get_fpr_table(results_files, fpr_tasks):
    fpr_list=[]
    for results_file in results_files:
        # print(results_file)
        fpr_table={}
        fpr_total=[]
        try:
            results = jload(results_file)
            try:
                opi_clean_pred=[]
                opi_clean_label=[]
                opi_tasks=['sms_spam', 'sst2', 'mrpc', 'hsol', 'rte', 'jfleg','gigaword']
                for target_task in opi_tasks:
                    opi_clean_pred.extend(results[f"opi_clean/{target_task}"])
                    opi_clean_label.extend(jload(f"TestData/opi_clean/{target_task}/label"))
                _, fpr, _ = get_evaluation_metrics(opi_clean_pred, opi_clean_label)
                fpr_table['opi'] = fpr
                fpr_total.append(fpr)
                # print(len(opi_clean_pred))
            except Exception as e:
                fpr_table['opi'] = ""

            other_tasks = [task for task in fpr_tasks if task != 'opi']
            for target_task in other_tasks:
                task_path=f"{target_task}_clean/{target_task}"
                if task_path in results:
                    pred=results[task_path]
                    label=jload(f"TestData/{task_path}/label")
                    _, fpr, _ = get_evaluation_metrics(pred, label)
                    fpr_table[target_task] = fpr
                    fpr_total.append(fpr)
                else:
                    fpr_table[target_task] = ""
                # print(len(pred))
        except Exception as e:
            for target_task in fpr_tasks:
                fpr_table[target_task] = ""
        if results_file.split('/')[-1].startswith('PIShield'):
            row_values=['{\\name} (Ours)']
        else:
            row_values=['']

        for task in fpr_tasks:
            if fpr_table[task] != "":
                row_values.append(f"{fpr_table[task]:.2f}")
            else:
                row_values.append("")
        # print(" & ".join(row_values) + " \\\\ \\hline")
        if any(value == "" for value in row_values):
            row_values.append("")
        else:
            numeric_values = [float(value) for value in row_values[1:] if value] 
            mean_value = np.mean(numeric_values)
            row_values.append(f"{mean_value:.3f}")

        print(" & ".join(row_values) + " \\\\ \\hline")
    #     fpr_list.append(round(np.mean(fpr_total),2))
    # print(f"FPR for all the results_files: {fpr_list}")
    return 

def get_fnr_table_per_dataset(results_files, dataset_name):
    print(f"\ndataset_name: {dataset_name}")
    fnr_list=[]
    for results_file in results_files:
        # print(results_file)
        fnr_total=[]
        fnr_table={}
        attack_strategies=['naive', 'escape', 'ignore', 'fake_comp', 'combine', 'universal', 'neural_exec', 'pleak']
        try:
            results = jload(results_file)      
            if dataset_name == 'opi':
                target_tasks=['sms_spam', 'sst2', 'mrpc', 'hsol', 'rte', 'jfleg','gigaword']
                inject_tasks=['sms_spam', 'sst2', 'mrpc', 'hsol', 'rte', 'jfleg','gigaword']    

                for attack_strategy in attack_strategies:
                    preds=[]
                    labels=[]
                    for target_task in target_tasks:
                        for inject_task in inject_tasks:
                            task = f"opi_malicious/end/{attack_strategy}/{target_task}_{inject_task}"
                            label=jload(f"TestData/{task}/label")
                            y_pred = results[task]
                            preds.extend(y_pred)
                            labels.extend(label)
                    _, _, fnr = get_evaluation_metrics(preds, labels)
                    fnr_table[attack_strategy] = fnr
                    fnr_total.append(fnr)
                    # print(len(preds))
            else:
                # this is the lookup table for the other datasets
                other_datasets={'dolly': ['dolly','dolly'],
                                'boolq': ['boolq','boolq'],
                                'mmlu': ['mmlu','mmlu'],
                                'hotelreview': ['hotelreview','close'],
                                }
                target_task, inject_task = other_datasets[dataset_name]

                for attack_strategy in attack_strategies:
                    task_path=f"{dataset_name}_malicious/end/{attack_strategy}/{target_task}_{inject_task}"
                    if task_path in results:
                        task=task_path
                        pred=results[task]
                        label=jload(f"TestData/{task}/label")
                        _, _, fnr = get_evaluation_metrics(pred, label)
                        fnr_table[attack_strategy] = fnr
                        fnr_total.append(fnr)
                    else:   
                        fnr_table[attack_strategy] = ""
                    # print(len(pred))
        except Exception as e:
            print(e)
            for attack_strategy in attack_strategies:
                fnr_table[attack_strategy] = ""
        if results_file.split('/')[-1].startswith('PIShield'):
            row_values=['{\\name} (Ours)']
        else:
            row_values=['']

        for attack_strategy in attack_strategies:
            if fnr_table[attack_strategy] != "":
                row_values.append(f"{fnr_table[attack_strategy]:.2f}")
            else:
                row_values.append("")

        # print(" & ".join(row_values) + " \\\\ \\hline")
        if any(value == "" for value in row_values):
            row_values.append("")
        else:
            numeric_values = [float(value) for value in row_values[1:] if value] 
            mean_value = np.mean(numeric_values)
            row_values.append(f"{mean_value:.3f}")
        print(" & ".join(row_values) + " \\\\ \\hline")
    #     fnr_list.append(round(np.mean(fnr_total),2))
    # print(f"FNR for all the results_files: {fnr_list}")
    return 




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_files", type=str, nargs='+', default=None)
    parser.add_argument("--results_file", type=str,  default=None)
    parser.add_argument("--target_task", type=str, default=None)
    parser.add_argument("--malicious_tasks", type=str, nargs='+', default=None)
    parser.add_argument("--cm_name", type=str, default=None)
    parser.add_argument("--test_data_name", type=str, default=None)
    parser.add_argument("--pos_strategy", type=str, default=None)
    parser.add_argument("--fpr_tasks", type=str, nargs='+', default=None)
    parser.add_argument("--fnr_tasks", type=str, nargs='+', default=None)
    
    parser.add_argument("--func", type=str, required=True)
    # parser.add_argument("--func", type=str, default="get_fnr_table_per_dataset")
    args = parser.parse_args()

    if args.func == "get_fpr_table":
        get_fpr_table(args.results_files, args.fpr_tasks)
    elif args.func == "get_fnr_table_per_dataset":
        get_fnr_table_per_dataset(args.results_files, args.test_data_name)
