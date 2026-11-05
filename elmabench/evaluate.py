"""ELMA anaphora (A2M) evaluation of a FlowMDM checkpoint.

    python -m elmabench.evaluate --flowmdm_dir FlowMDM \
        --model_path FlowMDM/results/humanml/ELMAv3_anap_hml3d_hist/model000550000.pt \
        --data_root /path/to/ELMA/HML3D --use_history_text --history_current_weight 0.5

Generates every sample of the evaluation file, keeps its target segment and compares it
against the GT segments of the test split with the T2M evaluator (FID, R-precision,
matching score, diversity), following FlowMDM's HumanML3D motion evaluation.
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

from elmabench.generated_data import GeneratedSegments, generate_targets, load_eval_samples
from elmabench.gt_data import GTSegments, GTTargets, load_segments, load_target_segments
from elmabench.upstream import enter_flowmdm

EVAL_MODES = {  # replications, diversity_times
    "debug": (1, 30),
    "fast": (5, 300),
    "final": (10, 300),
}
EVAL_BATCH_SIZE = 32  # R-precision is computed within batches of 32
SCENARIO = "anaphora"
EVAL_FILE = "humanml_test_set_anaphora.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flowmdm_dir", default="FlowMDM", help="clone of upstream FlowMDM (with assets)")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_root", required=True, help="ELMA HML3D directory (v1.0 or v1.1)")
    parser.add_argument("--eval_file", default="", help=f"defaults to <data_root>/{EVAL_FILE}")
    parser.add_argument("--eval_mode", default="final", choices=sorted(EVAL_MODES))
    parser.add_argument("--guidance_param", default=2.5, type=float)
    parser.add_argument("--use_history_text", action="store_true", help="A2M conditioning (history-text mix)")
    parser.add_argument("--history_current_weight", default=0.5, type=float)
    parser.add_argument("--generation_batch_size", default=128, type=int)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--output_dir", default="", help="defaults to <model dir>/elma_evaluation")
    parser.add_argument("--ground_truth", action="store_true",
                        help="score the real target segments instead of generated ones (GT reference row); "
                             "the model is still loaded so that the GT statistics match the model runs")
    args = parser.parse_args()
    if not 0.0 <= args.history_current_weight <= 1.0:
        parser.error("--history_current_weight must be in [0, 1]")
    for name in ["flowmdm_dir", "model_path", "data_root", "eval_file", "output_dir"]:
        if getattr(args, name):
            setattr(args, name, os.path.abspath(getattr(args, name)))
    args.eval_file = args.eval_file or os.path.join(args.data_root, EVAL_FILE)
    return args


def flowmdm_args(args):
    """Model/diffusion arguments resolved by FlowMDM's own evaluation parser from args.json."""
    from utils.parser_util import evaluation_parser
    argv = sys.argv
    sys.argv = ["eval", "--model_path", args.model_path, "--dataset", "humanml",
                "--guidance_param", str(args.guidance_param), "--seed", str(args.seed),
                "--device", str(args.device), "--eval_mode", args.eval_mode]
    try:
        return evaluation_parser()
    finally:
        sys.argv = argv


def log_name(args):
    model = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")
    cond = f"hist_w{args.history_current_weight:g}" if args.use_history_text else "nohist"
    data = os.path.basename(os.path.splitext(args.eval_file)[0])
    if args.ground_truth:
        return f"GT_{SCENARIO}_{data}_{args.eval_mode}_s{args.seed}"
    return f"{model}_{niter}_{SCENARIO}_{cond}_g{args.guidance_param:g}_{data}_{args.eval_mode}_s{args.seed}"


def main():
    args = parse_args()
    enter_flowmdm(args.flowmdm_dir)

    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data._utils.collate import default_collate
    from utils.fixseed import fixseed
    from utils import dist_util
    from utils.model_util import load_model
    from utils.metrics import evaluate_matching_score, evaluate_diversity, get_metric_statistics
    from data_loaders.humanml.utils.metrics import calculate_activation_statistics, calculate_frechet_distance
    from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
    from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
    from data_loaders.humanml.data.dataset import process_tokens
    from data_loaders.amass.babel import get_tokens

    def collate(batch):  # sorted by caption length, as the T2M text encoder expects
        batch.sort(key=lambda item: item[3], reverse=True)
        return default_collate(batch)

    replications, diversity_times = EVAL_MODES[args.eval_mode]
    out_dir = args.output_dir or os.path.join(os.path.dirname(args.model_path), "elma_evaluation")
    os.makedirs(out_dir, exist_ok=True)
    name = log_name(args)

    # The order below fixes the RNG stream: model init, evaluator init, GT batches, generation.
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    model, diffusion = load_model(flowmdm_args(args), dist_util.dev())
    eval_wrapper = EvaluatorMDMWrapper("humanml", dist_util.dev())

    w_vectorizer = WordVectorizer("./glove", "our_vab")
    eval_mean, eval_std = np.load("./dataset/t2m_mean.npy"), np.load("./dataset/t2m_std.npy")
    model_mean = np.load(os.path.join(args.data_root, "Mean.npy"))
    model_std = np.load(os.path.join(args.data_root, "Std.npy"))
    gt_dataset = GTSegments(load_segments(args.data_root), eval_mean, eval_std, w_vectorizer, process_tokens)
    gt_loader = DataLoader(gt_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=True, num_workers=0,
                           drop_last=False, collate_fn=collate)
    samples = load_eval_samples(args.eval_file, SCENARIO)
    print(f"GT segments: {len(gt_dataset)}  evaluation samples: {len(samples)}  ({args.eval_file})")

    log = open(os.path.join(out_dir, name + ".log"), "w")

    def report(line):
        print(line)
        print(line, file=log, flush=True)

    with torch.no_grad():
        gt_embeddings = [eval_wrapper.get_motion_embeddings(motions=batch[4], m_lens=batch[5]).cpu().numpy()
                         for batch in gt_loader]
    gt_mu, gt_cov = calculate_activation_statistics(np.concatenate(gt_embeddings, axis=0))

    metrics = {"MatchingScore": [], "R_precision": [], "FID": [], "Diversity": []}
    history_weight = args.history_current_weight if args.use_history_text else None
    for rep in range(replications):
        report(f"==================== Replication {rep} ====================")
        report(f"Time: {datetime.now()}")
        if args.ground_truth:
            generated = GTTargets(load_target_segments(args.data_root, samples), eval_mean, eval_std,
                                  w_vectorizer, process_tokens, get_tokens)
        else:
            targets = generate_targets(model, diffusion, samples, args.guidance_param, history_weight,
                                       args.generation_batch_size)
            generated = GeneratedSegments(targets, model_mean, model_std, eval_mean, eval_std,
                                          w_vectorizer, process_tokens, get_tokens)
        loader = DataLoader(generated, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0,
                            drop_last=False, collate_fn=collate)
        matching, r_precision, activations = evaluate_matching_score(eval_wrapper, {"motion": loader}, log)
        mu, cov = calculate_activation_statistics(activations["motion"])
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        report(f"---> [motion] FID: {fid:.4f}")
        diversity = evaluate_diversity(activations, log, diversity_times)
        metrics["MatchingScore"].append(matching["motion"])
        metrics["R_precision"].append(r_precision["motion"])
        metrics["FID"].append(fid)
        metrics["Diversity"].append(diversity["motion"])

    summary = {}
    for metric, values in metrics.items():
        mean, conf = get_metric_statistics(np.array(values), replications)
        report(f"========== {metric} Summary ==========")
        if np.ndim(mean) == 0:
            report(f"---> [motion] Mean: {mean:.4f} CInterval: {conf:.4f}")
            summary[metric] = [float(mean), float(conf)]
        else:
            report("---> [motion]" + "".join(f"(top {i + 1}) Mean: {m:.4f} CInt: {c:.4f};"
                                           for i, (m, c) in enumerate(zip(mean, conf))))
            for i, (m, c) in enumerate(zip(mean, conf)):
                summary[f"{metric}_top{i + 1}"] = [float(m), float(c)]
    log.close()
    summary["config"] = {key: value for key, value in vars(args).items()}
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {os.path.join(out_dir, name + '.json')}")


if __name__ == "__main__":
    main()
