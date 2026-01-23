import argparse
import time
from pishield import Extractor
from pishield import PIShield
from utils import jdump, jload

def test(args):
    print(args)
    results={}
    if args.detector_name == "PIShield":
        extractor = Extractor(args.model_name, args.format_id, args.token_position, batch_size=1)
        detector = PIShield(extractor)
        detector.load_probe(args.probe_name)
        time0=time.time()
        for data_name in args.test_datasets:
            time_start = time.time()
            print("-"*100)
            print(f"Data_name: {data_name}\n")
            y_pred = detector.test(data_name, args.layer_id, args.threshold)
            print(f"y_pred: {y_pred}")
            results[data_name] = y_pred
            jdump(results, args.output_name, indent_flag=0)
            time_end = time.time()
            print(f"Time taken: {time_end - time_start} seconds")
        time1=time.time()
        print(f"Total time taken: {time1 - time0} seconds")
    else:
        raise ValueError(f"Detector {args.detector_name} not supported")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector_name", type=str, default="PIShield")
    parser.add_argument("--test_datasets", nargs='+', default=[])
    parser.add_argument("--model_name", type=str, default="llama3-8b")
    parser.add_argument("--format_id", type=int, default=1)
    parser.add_argument("--token_position", type=str, default="last")
    parser.add_argument("--layer_id", type=int, default=12)
    parser.add_argument("--probe_name", type=str, default="data_llama3-8b_1_last/12")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_name", type=str, default="test_results")
    args = parser.parse_args()
    test(args)
