<div align="center">

# CKG-RAG: Enhancing LLM Reasoning in Medical Domain Using Contextual Knowledge Graphs


</div>

This includes the original implementation of CKG-RAG: Enhancing LLM Reasoning in Medical Domain Using Contextual Knowledge Graphs.

**D²-RAG** is a framework that achieves adaptive knowledge retrieval and utilization through a dual-stage decision mechanism.

## 📌 Abstract

Retrieval-augmented generation (RAG) has achieved significant progress in the medical question answering (MQA) domain. However, introducing multiple relevant passages in the retrieval stage frequently leads to excessively long input sequences. In such cases, large language models (LLMs) tend to focus on the beginning and end of the context while neglecting the middle content, a phenomenon known as lost in the middle, which ultimately undermines answer accuracy. To address this issue, we proposed CKG-RAG, a novel framework based on knowledge graph (KG) and soft prompting. First, we organize the retrieved documents into a structured KG to reduce semantic redundancy. Second, unlike conventional approaches that flatten graphs into natural language sequences, we adopt a soft prompting mechanism that incorporates graph-structured information into learnable prompt vectors, guiding the model toward structure-aware reasoning. We validate the effectiveness and advancement of our framework on multiple public MQA datasets.

## 🖼️ Framework

<p align="center">
  <img src="figure/framework.svg" alt="CKG-RAG Framework" width="90%">
</p>



## 📋 Content
1. [⚙️ Installation](#installation)
2. [🚀 Quick Start](#quick-start)
3. [📊 Baselines](#baselines)


## ⚙️ Installation
You can create a conda environment by running the command below.

```bash
conda env create -f environment.yml
```

## 🚀 Quick start
We provide [example data](example_data.jsonl). You can get our final results by by running the command below.

```bash
python example.py
```

📝 Your input file should be a `jsonl`.

[example.ipynb](example.ipynb) contains the complete implementation of our pipeline.

```bash
run example.ipynb
```

we use Qwen3-Embedding-4B as our embedding model. 

📚Medical textbook data coming soon.

[get_context_for_each_query_V2.py](get_context_for_each_query_V2.py) — Retrieves relevant documents for each query, powered by [LlamaIndex](https://www.llamaindex.ai/).

```bash
python get_context_for_each_query_V2.py
```

## 📊 Baselines

Implementation code for a subset of baseline methods.

Retrieval-Augmented Generation baseline.

```bash
run RAG.ipynb
```

Context-Aware Decoding (CAD) baseline.

```bash
run cad.ipynb
```

Decoding by Contrasting Layers (DoLa) baseline.

```bash
run dola.ipynb
```









