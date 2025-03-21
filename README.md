
## 👾 DIGIMON: Deep Analysis of Graph-Based Retrieval-Augmented Generation (RAG) Systems
<!-- <img src="img.png" alt="Description of the image" width="450" height="350"> -->

![Static Badge](https://img.shields.io/badge/DIGIMON-red)
![Static Badge](https://img.shields.io/badge/LLM-red)
![Static Badge](https://img.shields.io/badge/Graph_RAG-red)
![Static Badge](https://img.shields.io/badge/Document_QA-green)
![Static Badge](https://img.shields.io/badge/Document_Summarization-green)

> **GraphRAG** is a popular 🔥🔥🔥 and powerful 💪💪💪 RAG system! 🚀💡 Inspired by systems like Microsoft's, graph-based RAG is unlocking endless possibilities in AI.

> Our project focuses on **modularizing and decoupling** these methods 🧩 to **unveil the mystery** 🕵️‍♂️🔍✨ behind them and share fun and valuable insights! 🤩💫


## Representative Methods

We select the following Graph RAG methods:

| Method | Description| Link | Graph Type|
| --- |--- |--- | :---: | 
| RAPTOR | ICLR 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2401.18059-b31b1b.svg)](https://arxiv.org/abs/2401.18059)  [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/parthsarthi03/raptor)| Tree |
| KGP | AAAI 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2308.11730-b31b1b.svg)](https://arxiv.org/abs/2308.11730)  [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/YuWVandy/KG-LLM-MDQA)| Passage Graph |
| DALK | EMNLP 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2405.04819-b31b1b.svg)](https://arxiv.org/abs/2405.04819) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/David-Li0406/DALK)| ER Graph |
| HippoRAG | NIPS 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2405.14831-b31b1b.svg)](https://arxiv.org/abs/2405.14831) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/OSU-NLP-Group/HippoRAG) | ER Graph |
| MedGraphRAG | Medical Domain | [![arXiv](https://img.shields.io/badge/arXiv-2408.04187-b31b1b.svg)](https://arxiv.org/abs/2408.04187) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/SuperMedIntel/Medical-Graph-RAG) | ER Graph |
| G-retriever | NIPS 2024  | [![arXiv](https://img.shields.io/badge/arXiv-2402.07630-b31b1b.svg)](https://arxiv.org/abs/2402.07630) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/XiaoxinHe/G-Retriever)| ER Graph |
| ToG | ICLR 2024  | [![arXiv](https://img.shields.io/badge/arXiv-2307.07697-b31b1b.svg)](https://arxiv.org/abs/2307.07697) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/IDEA-FinAI/ToG)| ER Graph |
| GraphCoT | ACL 2024 | [![arXiv](https://img.shields.io/badge/arXiv-2404.07103-b31b1b.svg)](https://arxiv.org/abs/2404.07103) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PeterGriffinJin/Graph-CoT)| ER Graph |
| MS GraphRAG | Microsoft Project |  [![arXiv](https://img.shields.io/badge/arXiv-2404.16130-b31b1b.svg)](https://arxiv.org/abs/2404.16130) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/microsoft/graphrag)| KG |
| FastGraphRAG | CircleMind Project  | [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/circlemind-ai/fast-graphrag)| KG |
| LightRAG | High Star Project  | [![arXiv](https://img.shields.io/badge/arXiv-2410.05779-b31b1b.svg)](https://arxiv.org/abs/2410.05779) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/HKUDS/LightRAG)| RKG |

##  Graph Types
Based on the entity and relation, we categorize the graph into the following types:

+ **Chunk Tree**: A tree structure formed by document content and summary.
+ **Passage Graph**: A relational network composed of passages, tables, and other elements within documents.
+ **ER Graph**: An Entity-Relation Graph, which contains only entities and relations, is commonly represented as triples.
+ **KG**: A Knowledge Graph, which enriches entities with detailed descriptions and type information.
+ **RKG**: A Rich Knowledge Graph, which further incorporates keywords associated with relations.

The criteria for the classification of graph types are as follows:

|Graph Attributes | Chunk Tree |Passage Graph | ER  | KG | RKG |
| --- |--- |--- |--- | --- | --- |
|Original Content| ✅|✅| ❌|❌|❌| 
|Entity Name| ❌|❌|✅|✅|✅|
|Entity Type| ❌| ❌| ❌|✅|✅|
|Entity Description|❌| ❌| ❌|✅|✅|
|Relation Name| ❌|❌|✅|❌|✅|
|Relation keyword|❌| ❌| ❌|❌|✅|
|Relation Description|❌| ❌| ❌|✅|✅|
|Edge Weight| ❌|❌|✅|✅|✅|

##  Operators in the Retrieve Stage 
> The retrieval stage lies the **key role** ‼️ in the entire GraphRAG process. ✨ The goal is to identify query-relevant content that supports the generation phase, enabling the LLM to provide more **accurate** responses.


💡💡💡 After thoroughly reviewing all implementations, we've distilled them into a set of **16** operators 🧩🧩. Each method then constructs its retrieval module by combining one or more of these operators 🧩.

### Five Types of Operators

> We classify the operators into five categories, each offering a different way to retrieve and structure relevant information from graph-based data.

#### 📄 Chunk Operators
> retrieve the most relevant text segments (chunks) related to the query.

                                      
| Name               | Description                                                    | Example Methods              |
|--------------------|----------------------------------------------------------------|------------------------------|
| **by_ppr**          | Uses Personalized PageRank to identify relevant chunks.         | HippoRAG                     |
| **by_relationship** | Finds chunks that contain specified relationships.       | LightRAG                     |
| **entity_occurrence** | Retrieves chunks where both entities of an edge frequently appear together. | Local Search for MS GraphRAG |



#### ⭕️ Entity Operators
> retrieve entities (e.g., people, places, organizations) that are most relevant to the given query.

| Name            | Description                                                                        | Example Methods            |
|-----------------|------------------------------------------------------------------------------------|----------------------------|
| **by_relationship** | Use key relationships to retrieve relevant entities | LightRAG                   |
| **by_vdb**       | Find entities by vector-database  | G-retriever、 MedicalRAG、RAPTOR、KGP |
| **by_agent**     | Utilizes LLM to find the useful entities| TOG                       |
| **by_ppr**       | Use PPR to retrieve entities | FastGraphRAG     |                  |



#### ➡️ Relationship Operators
> extracting useful relationships for the given query.

| Name            | Description                                                                        | Example Methods            |
|-------------|-------------------------------------------------|--------------------------------------------------------|
| **by_vdb**         | Retrieve relationships by vector-database      | LightRAG、G-retriever |
| **by_agent**        | Utilizes LLM to find the useful entities| TOG       |
| **by_entity**       | One-hot neighbors of the key entities                         | Local Search for MS GraphRAG                          |
| **by_ppr**       | Use  PPR to retrieve relationships | FastGraphRAG     |                  |


#### 🔗 Community Operators
> Identify high-level information, which is only used for MS GraphRAG.


| **Name** | **Description**                                      | **Example Methods**                                  |
|----------------------|-----------------------------------------------------|-------------------------------------------------|
| **by_entity**        | Detects communities containing specified entities   | Local Search for MS GraphRAG                   |
| **by_level**         | Returns all communities below a specified level     | Global Search for MS GraphRAG                  | 


#### 📈 Subgraph Operators
> Extract a relevant subgraph for the given query

| Name              | Description                                                             | Example Methods |
|-------------------|-------------------------------------------------------------------------|-----------------|
| **by_path**        | Retrieves a path | DALK            |
| **by_Steiner Tree** | Constructs a minimal connecting subgraph (Steiner tree)  | G-retriever     |
| **induced_subgraph** | Extracts a subgraph induced by a set of entities and relationships.           | TOG             |



You can freely 🪽 combine those operators 🧩 to create more and more GraphRAG methods. 


#### 🌰 Examples 
> Below, we present some examples illustrating how existing algorithms leverage these operators.


| Name             | Operators                                                                                           |
|------------------|-----------------------------------------------------------------------------------------------------|
| **HippoRAG**     | Chunk (by_ppr)                                                                                      |
| **LightRAG**     | Chunk (by_relationship) + Entity (by_relationship) + Relationship (by_vdb)                          |
| **FastGraphRAG** | Chunk (by_ppr) + Entity (by_ppr) + Relationship (by_ppr)                                           |
