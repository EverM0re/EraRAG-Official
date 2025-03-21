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



def check_dirs(opt):
    # For each query, save the results in a separate directory
    result_dir = os.path.join(opt.working_dir, opt.exp_name, "Results")
    # Save the current used config in a separate directory
    config_dir = os.path.join(opt.working_dir, opt.exp_name, "Configs")
    # Save the metrics of entire experiment in a separate directory
    metric_dir = os.path.join(opt.working_dir, opt.exp_name, "Metrics")
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
    dataset_len = 100
    
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

    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
    parser.add_argument("-dataset_name", type=str, help="Name of the dataset.")
    args = parser.parse_args()

    opt = Config.parse(Path(args.opt), dataset_name=args.dataset_name)
    digimon = GraphRAG(config=opt)
    result_dir = check_dirs(opt)

    query_dataset = RAGQueryDataset(
        data_dir=os.path.join(opt.data_root, opt.dataset_name)
    )
    corpus = query_dataset.get_corpus()
    # corpus = corpus[:10]

    asyncio.run(digimon.insert(corpus))

    save_path = wrapper_query(query_dataset, digimon, result_dir)

    asyncio.run(wrapper_evaluation(save_path, opt, result_dir))

    # for train_item in dataloader:

    # a = asyncio.run(digimon.query("Who is Fred Gehrke?"))

    # asyncio.run(digimon.query("Who is Scrooge?"))
