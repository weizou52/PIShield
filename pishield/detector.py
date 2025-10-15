import os
import random
from utils import *
from pishield.utils import *


class PIShield:
    def __init__(self, extractor):
        self.train_data_dir = "TrainData"
        self.test_data_dir = "TestData"
        self.val_data_dir = "ValData"
        self.probes_dir = "probes"
        self.analysis_dir = "analysis"
        self.extractor=extractor
        self.linear_probe = None
        self.layer_ids = self.extractor.layer_ids

    def load_probe(self, probe_name):
        self.linear_probe = load_pickle(os.path.join(self.probes_dir, probe_name))
        print(f"\nLoaded probe {probe_name}\n")


    def train_linear_probe(self, data_name, layer_id, save_hidden_states=True, save_probe=True, save_analysis=True):
        print(f"train linear probe...")
        try:
            hs = load_pickle(f"{self.train_data_dir}/{data_name}/hs_{self.extractor.name}")
        except:
            hs=self.extractor(jload(f"{self.train_data_dir}/{data_name}/data"))
            if save_hidden_states:
                save_pickle(hs, f"{self.train_data_dir}/{data_name}/hs_{self.extractor.name}")
        label=jload(f"{self.train_data_dir}/{data_name}/label")

        # get hidden states for the specified layer
        data = hs[layer_id]
        # Shuffle and split data and label
        combined = list(zip(data, label))
        random.seed(42)
        random.shuffle(combined)
        data, label = zip(*combined)
        data = list(data)  
        label = list(label)
        split_idx = int(0.8* len(data))
        # train logistic regression model
        log_model, train_accuracy, test_accuracy = create_logistic_regression_model(data, label, split_idx)
        print(f"len(data): {len(data)}")
        print(f"train_accuracy: {train_accuracy}")
        print(f"test_accuracy: {test_accuracy}")
        if save_probe:
            save_pickle(log_model, f"{self.probes_dir}/{data_name}_{self.extractor.name}_{layer_id}")

        return train_accuracy, test_accuracy

    def train_linear_probe_all_layers(self, data_name, save_hidden_states=True, save_probe=True, save_analysis=True):
        print(f"train linear probe...")
        try:
            hs = load_pickle(f"{self.train_data_dir}/{data_name}/hs_{self.extractor.name}")
        except:
            hs=self.extractor(jload(f"{self.train_data_dir}/{data_name}/data"))
            if save_hidden_states:
                save_pickle(hs, f"{self.train_data_dir}/{data_name}/hs_{self.extractor.name}")
        label=jload(f"{self.train_data_dir}/{data_name}/label")
        for layer_id in self.layer_ids:
            # get hidden states for the specified layer
            data = hs[layer_id]
            # Shuffle and split data and label
            combined = list(zip(data, label))
            random.seed(42)
            random.shuffle(combined)
            t_data, t_label = zip(*combined)
            t_data = list(t_data)  
            t_label = list(t_label)
            split_idx = int(0.8* len(t_data))
            # train logistic regression model
            log_model, train_accuracy, test_accuracy = create_logistic_regression_model(t_data, t_label, split_idx)
            print(f"Layer {layer_id}:")
            print(f"train_accuracy: {train_accuracy}")
            print(f"test_accuracy: {test_accuracy}")
            if save_probe:
                save_pickle(log_model, f"{self.probes_dir}/{data_name}_{self.extractor.name}/{layer_id}")

    def test(self, data_name, layer_id, threshold=0.5):
        print(f"testing...")
        if self.linear_probe is None:
            print("Probe model does not exist")
            return None
        try:
            hs = load_pickle(f"{self.test_data_dir}/{data_name}/hs_{self.extractor.name}")
        except:
            hs=self.extractor(jload(f"{self.test_data_dir}/{data_name}/data"))
            save_pickle(hs, f"{self.test_data_dir}/{data_name}/hs_{self.extractor.name}")
        data = hs[layer_id]
        y_pred = get_log_predictions(log_model = self.linear_probe, X_test = data, threshold=threshold)
        return y_pred
    
    def predict(self, examples, layer_id, threshold=0.5):
        print(f"predicting...")
        if self.linear_probe is None:
            print("Probe model does not exist")
            return None
        hs=self.extractor(examples)[layer_id]
        y_prob = self.linear_probe.predict_proba(hs)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        return y_prob, y_pred

    def evaluate(self, y_pred, labels):
        conf_matrix_log, accuracy, fpr, fnr = get_evaluation_metrics(y_pred, labels)
        print(f"conf_matrix: {conf_matrix_log}")
        print(f"accuracy: {accuracy}")
        print(f"fpr: {fpr}")
        print(f"fnr: {fnr}")
        return conf_matrix_log, accuracy, fpr, fnr


