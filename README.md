



<h1 align="center"> 💫 EraRAG: Efficient and Incremental Retrieval-Augmented Generation for Growing Corpora </a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<h5 align="center">

<img src="figures/main1.png">

</h5>

This is the official implementation of the following paper: 
> **EraRAG: Efficient and Incremental Retrieval-Augmented Generation for Growing Corpora** [[Paper](https://arxiv.org/abs/2506.20963)]
> 
> Fangyuan Zhang, Zhengjun Huang, Yingli Zhou, Qintian Guo, Zhixun Li, Wensheng Luo, Di Jiang, Yixiang Fang, Xiaofang Zhou

EraRAG is a novel hierarchical graph construction framework that supports dynamic updates through localized selective re-partitioning, enabling efficient and scalable retrieval with strong static accuracy and stable performance under corpus changes.

## 💫 Key Features

- **Accuracy**: Achieves state-of-the-art performance across diverse question types, including multi-hop and long-document QA tasks.
- **Efficiency**: Significantly reduces both graph construction time and token consumption compared to existing RAG baselines.
- **Incremental Updates**: Supports fast and efficient integration of new documents without requiring global tree reconstruction, enabling dynamic corpus adaptation.


## 🚀 Get Start

EraRAG and controled baselines are built on the unified framework proposed by [In-depth study of graphrag](https://github.com/JayLZhou/GraphRAG). To run EraRAG, use the following command:
```
python main.py -opt <Method>.yaml -dataset_name <Datasetname> -external_tree <External tree path> -root <rootname> -query <wether to query>
```

On default, EraRAG will treat the input corpus as new corpus and enforce a global reconstruction. To make a insertion to a existing tree, set Dynamic.yaml key parameters as follows.

```
force: False
add: True
```


<!-- If you want to try reproducing the baseline methods, simply run:

```
bash run_baseline.sh
```

If you want to try reproducing the performance of IceBerg, simply run:

```
bash run_iceberg.sh
``` -->

## 🧰 Experimental Settings

We have incorporated several baseline methods and benchmark datasets:

| Baseline |Paper| Code |
| -------- |-| ---- |
| GraphRAG       |[From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130#) | [graphrag](https://github.com/microsoft/graphrag) |
| LightRAG       |[LightRAG: Simple and Fast Retrieval-Augmented Generation](https://arxiv.org/abs/2410.05779) | [LightRAG](https://github.com/HKUDS/LightRAG) |
| HippoRAG       |[HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://arxiv.org/abs/2405.14831)| [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)    |
| RAPTOR       |[RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)| [raptor](https://github.com/parthsarthi03/raptor)    |



## ⚙️ Experimental Results

Our proposed EraRAG framework achieves significant retrieval performance against state of the art graph-based RAG frameworks.

<img src="figures/static.png">

Thanks to the proposed selective reconstruction mechanism, EraRAG is able to perform fast insertions on evolving corpora, surpassing benchmarks on time and token cost reduction.

<img src="figures/dynamic.png">

## Acknowledgements

We acknowledge these excellent works for providing open-source code: [GraphRAG](https://github.com/microsoft/graphrag), [RAPTOR](https://github.com/parthsarthi03/raptor), [LightRAG](https://github.com/HKUDS/LightRAG), [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG), [In-depth study of graphrag](https://github.com/JayLZhou/GraphRAG).


## 🤗 Citation
Please consider citing our work if you find it helpful:
```
@article{zhang2025erarag,
      title={EraRAG: Efficient and Incremental Retrieval Augmented Generation for Growing Corpora}, 
      author={Fangyuan Zhang and Zhengjun Huang and Yingli Zhou and Qintian Guo and Zhixun Li and Wensheng Luo and Di Jiang and Yixiang Fang and Xiaofang Zhou},
      journal={arXiv preprint arXiv:2506.20963},
      year={2025} 
}
```