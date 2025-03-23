from typing import Optional

from Core.Utils.YamlModel import YamlModel


class GraphConfig(YamlModel):
    graph_type: str = "er_graph"
    # Building graph
    extract_two_step: bool = True
    max_gleaning: int = 1
    force: bool = False
    add: bool = False

    # For ER graph & KG graph and & RKG graph
    enable_entity_description: bool = False
    enable_entity_type: bool = False
    enable_edge_description: bool = False
    enable_edge_name: bool = False
    prior_prob: float = 0.8
    enable_edge_keywords: bool = False
    # Graph clustering
    use_community: bool = False  # Default to False
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF
    summary_max_tokens: int = 500
    llm_model_max_token_size: int = 32768

    # For Tree graph config 
    build_tree_from_leaves: bool = False
    reduction_dimension: int = 5
    summarization_length: int = 100
    num_layers: int = 10
    top_k: int = 5
    threshold_cluster_num: int = 5000
    start_layer: int = 5
    graph_cluster_params: Optional[dict] = None
    selection_mode: str = "top_k"
    max_length_in_cluster: int = 3500
    threshold: float = 0.1
    cluster_metric: str = "cosine"
    verbose: bool = False
    random_seed: int = 224
    enforce_sub_communities: bool = False
    max_size_percentage: float = 0.2
    tol: float = 1e-4
    max_iter: int = 300
    size_of_clusters: int = 10

    # LSH clustering config
    upper_limit: int = 15
    lower_limit: int = 10
    num_hyperplanes: int = 20

    # For graph augmentation
    similarity_threshold: float = 0.8
    similarity_top_k: int = 10
    similarity_max: float = 1.0
