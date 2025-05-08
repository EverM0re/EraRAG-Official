from Core.GraphRAG import GraphRAG
from Option.Config2 import Config
import argparse
import os
import asyncio
from pathlib import Path
from shutil import copyfile
from Data.QueryDataset import RAGQueryDataset
import pandas as pd
from Core.Utils.Evaluation import Evaluator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
    parser.add_argument("-dataset_name", type=str, help="Name of the dataset.")
    parser.add_argument("-external_tree", type=str, help="Path to external tree file to load from.")
    parser.add_argument("-root", type=str, default="", help="Root directory to prefix result/config/metric paths.")
    parser.add_argument("-query", type=str, default="0", help="Whether to run query and evaluation (1 to enable).")
    return parser.parse_args()

def check_dirs(opt, root):
    base_dir = os.path.join(opt.working_dir, opt.exp_name, root) if root else os.path.join(opt.working_dir, opt.exp_name)

    result_dir = os.path.join(base_dir, "Results")
    config_dir = os.path.join(base_dir, "Configs")
    metric_dir = os.path.join(base_dir, "Metrics")

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)

    opt_name = args.opt[args.opt.rindex("/") + 1 :]
    basic_name = os.path.join(args.opt.split("/")[0], "Config2.yaml")

    copyfile(args.opt, os.path.join(config_dir, opt_name))
    copyfile(basic_name, os.path.join(config_dir, "Config2.yaml"))

    return result_dir

def wrapper_query(query_dataset, digimon, result_dir):
    all_res = []

    dataset_len = len(query_dataset)
    dataset_len = 4609

    for _, i in enumerate(range(dataset_len)):
        query = query_dataset[i]
        res = asyncio.run(digimon.query(query["question"]))
        query["output"] = res
        all_res.append(query)

    all_res_df = pd.DataFrame(all_res)
    save_path = os.path.join(result_dir, "results.json")
    all_res_df.to_json(save_path, orient="records", lines=True)
    return save_path

async def wrapper_evaluation(path, opt, result_dir):
    eval = Evaluator(path, opt.dataset_name)
    res_dict = await eval.evaluate()
    save_path = os.path.join(result_dir, "metrics.json")
    with open(save_path, "w") as f:
        f.write(str(res_dict))

if __name__ == "__main__":
    args = parse_args()

    opt = Config.parse(Path(args.opt), dataset_name=args.dataset_name)
    digimon = GraphRAG(config=opt)
    result_dir = check_dirs(opt, args.root)

    if args.external_tree and args.external_tree.strip():
        digimon.graph._graph.set_external_tree_path(args.external_tree.strip())

    query_dataset = RAGQueryDataset(
        data_dir=os.path.join(opt.data_root, opt.dataset_name)
    )
    corpus = query_dataset.get_corpus()

    asyncio.run(digimon.insert(corpus))

    if args.query == "1":
        save_path = wrapper_query(query_dataset, digimon, result_dir)
        asyncio.run(wrapper_evaluation(save_path, opt, result_dir))
