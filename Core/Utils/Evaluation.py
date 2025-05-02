import json
import pandas as pd
import re
import string
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
import copy
import numpy as np
import mauve
import nltk
from nltk import sent_tokenize
from rouge_score import rouge_scorer, scoring
from Option.Config2 import default_config
from Core.Provider.BaseLLM import BaseLLM
from Core.Provider.LLMProviderRegister import create_llm_instance
from Core.Common.Logger import logger


# Path to your downloaded tokenizer
nltk_path = "/ssddata/zhengjun/NLTK"

# add the nltk_path to the nltk.data.path
nltk.data.path.append(nltk_path)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print("Downloading punkt")
    nltk.download("punkt", download_dir=nltk_path)
    nltk.download("wordnet", download_dir=nltk_path)


def bleu_1(p, g):
    return sentence_bleu(g, p, weights=(1, 0, 0, 0))


def bleu_4(p, g):
    return sentence_bleu(g, p, weights=(0, 0, 0, 1))


def bleu_4_modify(p, g):
    return sentence_bleu(g, p, weights=(0.25, 0.25, 0.25, 0.25))


def bleu_1_smooth(p, g):
    return sentence_bleu(
        g, p, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1
    )


def bleu_4_smooth(p, g):
    return sentence_bleu(
        g, p, weights=(0, 0, 0, 1), smoothing_function=SmoothingFunction().method1
    )


def bleu_4_modify_smooth(p, g):
    return sentence_bleu(
        g,
        p,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=SmoothingFunction().method1,
    )


def meteor(p, g):
    return meteor_score([x.split() for x in g], p.split())


# 创建 RougeScorer 实例，设置 ROUGE-L 指标
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


# 计算 ROUGE-L 分数
def rouge_l(p, g):
    if isinstance(g, list):
        g = g[0]

    return scorer.score(g, p)  # g: ground truth, p: prediction


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, tokenize=False):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        if tokenize:
            score = metric_fn(word_tokenize(prediction), [word_tokenize(ground_truth)])
        else:
            score = metric_fn(prediction, [ground_truth])
        scores_for_ground_truths.append(score)

    if isinstance(score, dict) and "rougeL" in score:
        rouge_l_score = {"rouge_l f1": 0, "rouge_l precision": 0, "rouge_l recall": 0}
        rouge_l_score["rouge_l f1"] = max(
            [score["rougeL"].fmeasure for score in scores_for_ground_truths]
        )
        rouge_l_score["rouge_l precision"] = max(
            [score["rougeL"].precision for score in scores_for_ground_truths]
        )
        rouge_l_score["rouge_l recall"] = max(
            [score["rougeL"].recall for score in scores_for_ground_truths]
        )

        return rouge_l_score
    else:
        return round(max(scores_for_ground_truths), 2)


def get_metric_score(prediction, ground_truths):
    bleu_1_score = metric_max_over_ground_truths(
        bleu_1, prediction, ground_truths, tokenize=True
    )
    bleu_4_score = metric_max_over_ground_truths(
        bleu_4, prediction, ground_truths, tokenize=True
    )
    modify_bleu_4_score = metric_max_over_ground_truths(
        bleu_4_modify, prediction, ground_truths, tokenize=True
    )
    bleu_1_smooth_score = metric_max_over_ground_truths(
        bleu_1_smooth, prediction, ground_truths, tokenize=True
    )
    bleu_4_smooth_score = metric_max_over_ground_truths(
        bleu_4_smooth, prediction, ground_truths, tokenize=True
    )
    modify_bleu_4_smooth_score = metric_max_over_ground_truths(
        bleu_4_modify_smooth, prediction, ground_truths, tokenize=True
    )
    meteor_score = metric_max_over_ground_truths(
        meteor, prediction, ground_truths, tokenize=False
    )
    rouge_l_score = metric_max_over_ground_truths(
        rouge_l, prediction, ground_truths, tokenize=False
    )

    return {
        "bleu_1": bleu_1_score,
        "bleu_4": bleu_4_score,
        "modify_bleu_4": modify_bleu_4_score,
        "bleu_1_smooth": bleu_1_smooth_score,
        "bleu_4_smooth": bleu_4_smooth_score,
        "modify_bleu_4_smooth": modify_bleu_4_smooth_score,
        "meteor": meteor_score,
        "rouge_l f1": rouge_l_score["rouge_l f1"],
        "rouge_l precision": rouge_l_score["rouge_l precision"],
        "rouge_l recall": rouge_l_score["rouge_l recall"],
    }


class Evaluator:
    def __init__(self, eval_path: str, dataset_name: str):

        self.path = eval_path
        self.config = default_config
        self.llm = create_llm_instance(self.config.llm)
        self.dataset_name = dataset_name
        self.short_eval_metrics = ["accuracy", "f1", "precision", "recall", "em"]
        self.close_eval_metrics = ["accuracy"]
        self.long_narrative_metrics = [
            "bleu_1",
            "bleu_4",
            "modify_bleu_4",
            "bleu_1_smooth",
            "bleu_4_smooth",
            "modify_bleu_4_smooth",
            "meteor",
            "rouge_l f1",
            "rouge_l precision",
            "rouge_l recall",
        ]
        self.long_asqa_metrics = ["str_em", "str_hit", "rougeLsum", "mauve"]

        self.dataset_mode_map = {
            "hotpotqa": "short-form",
            "multihop-rag": "short-form",
            "Popqa": "short-form",
            "ALCE": "long-asqa",
            "quality": "close-set",
        }
        if "narrative" in dataset_name:
            self.mode = "long-narrative"
        else:
            self.mode = self.dataset_mode_map.get(dataset_name, "short-form")

    async def evaluate(self):
        df = pd.read_json(self.path, lines=True)
        logger.info(f"Loaded {len(df)} records from {self.path}")
        logger.info(f"Evaluating {self.mode} mode.")

        if self.mode == "short-form":
            self.print_eval_matrics(self.short_eval_metrics)
            res_dict, df = self.short_eval(df)

        elif self.mode == "long-narrative":
            self.print_eval_matrics(self.long_narrative_metrics)
            res_dict, df = self.long_narrative_eval(df)

        elif self.mode == "long-asqa":
            self.print_eval_matrics(self.long_asqa_metrics)
            res_dict, df = self.long_asqa_eval(df)

        elif self.mode == "close-set":
            self.print_eval_matrics(self.close_eval_metrics)
            res_dict, df = await self.close_eval(df)

        else:
            raise ValueError("Invalid evaluation mode.")

        # add .score to the save path, before the .json
        save_path = self.path.replace(".json", ".score.json")
        df.to_json(save_path, orient="records", lines=True)
        return res_dict

    def print_eval_matrics(self, eval_matrics):
        logger.info("In this evaluation, the following metrics are used:")
        metrics_str = " ".join(eval_matrics)
        logger.info(metrics_str)

    def get_label_pred_list(self, df, pred_col, label_col):
        label_list = df[label_col].tolist()
        pred_list = df[pred_col].tolist()
        return label_list, pred_list

    def short_eval(self, df: pd.DataFrame):
        # Short form evaluation code is referenced from the HippoRAG evaluation script:
        # links: https://github.com/OSU-NLP-Group/HippoRAG

        # Load results
        accuracy_list = []
        f1_list = []
        precission_list = []
        recall_list = []
        em_list = []

        label_list, pred_list = self.get_label_pred_list(df, "output", "answer")

        for prediction, answer in zip(pred_list, label_list):
            prediction = prediction.replace("|", "\n")
            prediction = prediction.split("\n")
            prediction_str = " ".join(prediction)

            answer = answer.split("|")
            if isinstance(answer, list):
                answer_str = " ".join(answer)
            else:
                answer_str = answer

            accuracy = self.eval_accuracy(prediction_str, answer_str)
            f1, prec, recall = self.f1_score(prediction_str, answer_str)
            em = self.exact_match_score(prediction_str, answer_str)
            em_list.append(em)
            f1_list.append(f1)
            precission_list.append(prec)
            recall_list.append(recall)
            accuracy_list.append(accuracy)

        accuracy = sum(accuracy_list) * 100 / len(accuracy_list)
        f1 = sum(f1_list) * 100 / len(f1_list)
        pre = sum(precission_list) * 100 / len(precission_list)
        recall = sum(recall_list) * 100 / len(recall_list)
        em = sum(em_list) * 100 / len(em_list)

        df["accuracy"] = accuracy_list
        df["f1"] = f1_list
        df["precision"] = precission_list
        df["recall"] = recall_list
        df["em"] = em_list

        res_dict = {
            "accuracy": accuracy,
            "f1": f1,
            "precision": pre,
            "recall": recall,
            "em": em,
        }

        logger.info(f"accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {pre:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1: {f1:.4f}")
        logger.info(f"EM: {em:.4f}")

        return res_dict, df

    def long_narrative_eval(self, df: pd.DataFrame):
        label_list, pred_list = self.get_label_pred_list(df, "output", "answer")

        # Load results
        bleu_1_list = []
        bleu_4_list = []
        modify_bleu_4_list = []
        bleu_1_smooth_list = []
        bleu_4_smooth_list = []
        modify_bleu_4_smooth_list = []
        meteor_list = []
        rouge_l_f1_list = []
        rouge_l_precision_list = []
        rouge_l_recall_list = []

        for prediction, answer in zip(pred_list, label_list):
            prediction = prediction.replace("|", "\n")
            prediction = prediction.split("\n")
            prediction_str = " ".join(prediction)

            metrics_res = get_metric_score(prediction_str, answer)
            bleu_1_list.append(metrics_res["bleu_1"])
            bleu_4_list.append(metrics_res["bleu_4"])
            modify_bleu_4_list.append(metrics_res["modify_bleu_4"])
            bleu_1_smooth_list.append(metrics_res["bleu_1_smooth"])
            bleu_4_smooth_list.append(metrics_res["bleu_4_smooth"])
            modify_bleu_4_smooth_list.append(metrics_res["modify_bleu_4_smooth"])
            meteor_list.append(metrics_res["meteor"])
            rouge_l_f1_list.append(metrics_res["rouge_l f1"])
            rouge_l_precision_list.append(metrics_res["rouge_l precision"])
            rouge_l_recall_list.append(metrics_res["rouge_l recall"])

        bleu_1 = sum(bleu_1_list) * 100 / len(bleu_1_list)
        bleu_4 = sum(bleu_4_list) * 100 / len(bleu_4_list)
        modify_bleu_4 = sum(modify_bleu_4_list) * 100 / len(modify_bleu_4_list)
        bleu_1_smooth = sum(bleu_1_smooth_list) * 100 / len(bleu_1_smooth_list)
        bleu_4_smooth = sum(bleu_4_smooth_list) * 100 / len(bleu_4_smooth_list)
        modify_bleu_4_smooth = (
            sum(modify_bleu_4_smooth_list) * 100 / len(modify_bleu_4_smooth_list)
        )
        meteor = sum(meteor_list) * 100 / len(meteor_list)
        rouge_l_f1 = sum(rouge_l_f1_list) * 100 / len(rouge_l_f1_list)
        rouge_l_precision = (
            sum(rouge_l_precision_list) * 100 / len(rouge_l_precision_list)
        )
        rouge_l_recall = sum(rouge_l_recall_list) * 100 / len(rouge_l_recall_list)

        df["bleu_1"] = bleu_1_list
        df["bleu_4"] = bleu_4_list
        df["modify_bleu_4"] = modify_bleu_4_list
        df["bleu_1_smooth"] = bleu_1_smooth_list
        df["bleu_4_smooth"] = bleu_4_smooth_list
        df["modify_bleu_4_smooth"] = modify_bleu_4_smooth_list
        df["meteor"] = meteor_list
        df["rouge_l_f1"] = rouge_l_f1_list
        df["rouge_l_precision"] = rouge_l_precision_list
        df["rouge_l_recall"] = rouge_l_recall_list

        logger.info(f"Bleu-1: {bleu_1:.4f}")
        logger.info(f"Bleu-4: {bleu_4:.4f}")
        logger.info(f"Modify Bleu-4: {modify_bleu_4:.4f}")
        logger.info(f"Bleu-1 Smooth: {bleu_1_smooth:.4f}")
        logger.info(f"Bleu-4 Smooth: {bleu_4_smooth:.4f}")
        logger.info(f"Modify Bleu-4 Smooth: {modify_bleu_4_smooth:.4f}")
        logger.info(f"Meteor: {meteor:.4f}")
        logger.info(f"Rouge-l F1: {rouge_l_f1:.4f}")
        logger.info(f"Rouge-l Precision: {rouge_l_precision:.4f}")
        logger.info(f"Rouge-l Recall: {rouge_l_recall:.4f}")

        res_dict = {
            "bleu_1": bleu_1,
            "bleu_4": bleu_4,
            "modify_bleu_4": modify_bleu_4,
            "bleu_1_smooth": bleu_1_smooth,
            "bleu_4_smooth": bleu_4_smooth,
            "modify_bleu_4_smooth": modify_bleu_4_smooth,
            "meteor": meteor,
            "rouge_l f1": rouge_l_f1,
            "rouge_l precision": rouge_l_precision,
            "rouge_l recall": rouge_l_recall,
        }

        return res_dict, df

    def long_asqa_eval(self, df: pd.DataFrame):
        # Long ASQA code is referenced from official ALCE repository:
        # links: https://github.com/princeton-nlp/ALCE
        
        str_em_list = []
        str_hit_list = []

        for index, row in df.iterrows():
            prediction = row["output"]
            answer = row["answer"]
            answer_pairs = row["qa_pairs"]
            annotations = row["annotations"]

            prediction = prediction.replace("|", "\n")
            prediction = prediction.split("\n")
            prediction_str = " ".join(prediction)
            prediction_str = self.normalize_answer(prediction_str)

            str_em, str_hit = self.eval_str_em(prediction_str, answer_pairs)
            str_em_list.append(str_em)
            str_hit_list.append(str_hit)

        mauve = self.compute_mauve(df)
        rougeLsum = self.compute_rouge(df)

        str_em = sum(str_em_list) * 100 / len(str_em_list)
        str_hit = sum(str_hit_list) * 100 / len(str_hit_list)
        # rougeLsum = sum(rougeLsum_list) / len(rougeLsum_list)

        df["str_em"] = str_em_list
        df["str_hit"] = str_hit_list
        df["rougeLsum"] = rougeLsum

        logger.info(f"str_em: {str_em:.4f}")
        logger.info(f"str_hit: {str_hit:.4f}")
        logger.info(f"mauve: {mauve:.4f}")
        logger.info(f"rougeLsum: {rougeLsum:.4f}")

        res_dict = {
            "str_em": str_em,
            "str_hit": str_hit,
            "mauve": mauve,
            "rougeLsum": rougeLsum,
        }

        return res_dict, df

    async def close_eval(self, df: pd.DataFrame):
        # Close set evaluation first use LLM to extract the option from the model output
        # Then, evaluate the extracted option with the answer index
        
    
        for index, row in df.iterrows():
            prompt = CLOSE_EXTRACT_OPTION_PORMPT.format(
                question=row["question"], model_output=row["output"]
            )
            response = await self.llm.aask(msg=prompt, format="json")

            try:
                df.loc[index, "extract_output"] = response["predict"]
            except Exception as e:
                df.loc[index, "extract_output"] = "-1"
        logger.info("LLM extract option completed.")

        accuracy_list = []
        label_list, pred_list = self.get_label_pred_list(
            df, "extract_output", "answer_idx"
        )

        for prediction, answer in zip(pred_list, label_list):
            prediction = prediction.strip()
            answer = answer.strip()
            accuracy = self.exact_match_score(prediction, answer)
            accuracy_list.append(accuracy)

        accuracy = sum(accuracy_list) * 100 / len(accuracy_list)
        df["accuracy"] = accuracy_list
        res_dict = {"accuracy": accuracy}
        logger.info(f"accuracy: {accuracy:.4f}")
        return res_dict, df

    def exact_presence(self, short_answers, context):
        """Verify if any of the answers is present in the given context.
        Args:
            short_answers: list of short answers to look for in the context
            context: a paragraph to search for short answers
        Returns:
            true if any of the short answers is present in the context
        """

        n_short_answers = [self.normalize_answer(sa) for sa in short_answers]
        n_context = self.normalize_answer(context)

        for ans in n_short_answers:
            if ans in n_context:
                return True

        return False

    def eval_str_em(self, prediction, qa_pairs: list):
        if len(qa_pairs) == 0:
            return 0, 0

        loc_acc = []
        for qa_pair in qa_pairs:
            loc_acc.append(self.exact_presence(qa_pair["short_answers"], prediction))

        acc = np.mean(loc_acc)
        hit = int(acc == 1)

        return acc, hit

    def compute_mauve(self, df):
        human_data = []
        model_data = []
        for idx, row in df.iterrows():
            # Remove ending punctuations
            # Remove any new lines
            # Truncate by 100 words
            human_data.append(
                " ".join(
                    (row["question"] + " " + row["answer"].strip()).split()[:100]
                ).rstrip(string.punctuation)
            )
            model_data.append(
                " ".join(
                    (row["question"] + " " + row["output"].strip()).split()[:100]
                ).rstrip(string.punctuation)
            )

        out = mauve.compute_mauve(
            p_text=human_data,
            q_text=model_data,
            device_id=0,
            max_text_length=512,
            verbose=True,
            batch_size=8,
            featurize_model_name="gpt2-large",
        )
        return out.mauve * 100

    def compute_rouge(self, df):
        def _rouge_calculation(
            hypotheses, references1, references2=[], metrics=["rougeLsum"]
        ):

            if references2 == []:
                references2 = references1

            scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
            aggregator = scoring.BootstrapAggregator()

            for i in range(len(hypotheses)):
                scores1 = scorer.score(references1[i], hypotheses[i])
                scores2 = scorer.score(references2[i], hypotheses[i])
                if scores1["rougeLsum"].fmeasure > scores2["rougeLsum"].fmeasure:
                    aggregator.add_scores(scores1)
                else:
                    aggregator.add_scores(scores2)

            scores = {m: [] for m in metrics}

            for m in metrics:
                fmeasure = aggregator.aggregate()[m].mid.fmeasure
                scores[m].append(fmeasure)

            for m in scores:
                scores[m] = 100 * sum(scores[m]) / len(scores[m])

            return scores

        hypotheses = {}
        references1 = {}
        references2 = {}

        for idx, item in df.iterrows():
            hypotheses[idx] = item["output"]
            if "annotations" in item and item["annotations"] is not None:  # For ASQA
                references1[idx] = item["annotations"][0]["long_answer"]
                references2[idx] = item["annotations"][1]["long_answer"]
            else:
                references1[idx] = item["answer"]
                references2[idx] = item["answer"]

        h, r1, r2 = [], [], []

        for key in references1:
            h.append(hypotheses[key])
            r1.append(references1[key])

            if references2 is not None:
                r2.append(references2[key])

        h = ["\n".join(sent_tokenize(text.lower())) for text in h]
        r1 = ["\n".join(sent_tokenize(text.lower())) for text in r1]
        r2 = ["\n".join(sent_tokenize(text.lower())) for text in r2]
        scores = _rouge_calculation(h, r1, r2)

        return scores["rougeLsum"]

    def normalize_answer(self, s):
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
            # return re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def f1_score(self, prediction, ground_truth):
        normalized_prediction = self.normalize_answer(prediction)
        normalized_ground_truth = self.normalize_answer(ground_truth)

        ZERO_METRIC = (0, 0, 0)

        if (
            normalized_prediction in ["yes", "no", "noanswer"]
            and normalized_prediction != normalized_ground_truth
        ):
            return ZERO_METRIC
        if (
            normalized_ground_truth in ["yes", "no", "noanswer"]
            and normalized_prediction != normalized_ground_truth
        ):
            return ZERO_METRIC

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return ZERO_METRIC
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, precision, recall

    def exact_match_score(self, prediction, ground_truth):
        return self.normalize_answer(prediction) == self.normalize_answer(ground_truth)

    def eval_accuracy(self, prediction: str, ground_truth: str):
        s1 = self.normalize_answer(prediction)
        s2 = self.normalize_answer(ground_truth)
        if s2 in s1:
            return 1
        else:
            return 0


CLOSE_EXTRACT_OPTION_PORMPT = """
You are given a model output which is a string. The model output is a list of options. You have to extract the option letter from the model output.

# GOAL

Your goal is to extract the option letter directly from the model output. You should not rely on any external knowledge or context to answer. Simply extract the option letter as stated in the model output.

# FORMAT

Please provide your answer in the following JSON format:

- ANSWER_OPTION: the option letter extracted from the model output.

    {{
        "model_output": <answer_option>
    }}

### Example 1
-----------
# INPUT:

Question:
How much time has passed between Blake's night with Eldoria and his search for Sabrina York in his mind-world?
A: 7 years
B: 10 hours
C: 12 years
D: 1 hour

# Model Output: 
I think the answer is 7 years.

OUTPUT:
    {{
        "predict": "A"
    }}

### Example 2
-----------
# INPUT:

Question:
How much time has passed between Blake's night with Eldoria and his search for Sabrina York in his mind-world?
A: 7 years
B: 10 hours
C: 12 years
D: 1 hour

# Model Output: 
The correct answer is C.

OUTPUT:
    {{
        "predict": "C"
    }}
    
### EXAMPLE 3
-----------

# INPUT:

Question:
Donald Trump is the president of:
A: China
B: Canada
C: France
D: Spain

# Model Output: 
The correct answer is: None of the above.

OUTPUT:
    {{
        "predict": "-1"
    }}

Now please the output based on the given question and model output.

### Real Data
# INPUT:

Question:
{question}

# Model Output:
{model_output}

OUTPUT:"""
