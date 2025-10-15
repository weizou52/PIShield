import argparse
import time
from pishield import Extractor
from pishield import PIShield

def train(args):
    print(args)
    time_start = time.time()
    extractor = Extractor(args.model_name, args.format_id, args.token_position)
    detector = PIShield(extractor)
    # detector.train_linear_probe(args.data_name, args.layer_id)
    detector.train_linear_probe_all_layers(args.data_name)
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama3-8b")
    parser.add_argument("--format_id", type=int, default=1)
    parser.add_argument("--token_position", type=str, default="last")
    parser.add_argument("--data_name", type=str, default="data")
    parser.add_argument("--layer_id", type=int, default=12)
    args = parser.parse_args()
    train(args)
