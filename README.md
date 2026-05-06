# ICL_RAG
### `ICL_equ_GD.ipynb`

This is the main notebook of the repository. It includes the core experiments and analysis for comparing in-context learning behavior with gradient-descent-style updates.

The notebook can be used to:

- run the main experiments;
- analyze the connection between ICL and gradient descent;
- study how retrieved documents interact with the query;
- inspect intermediate results and visualizations.

## Supporting Files

### `0_GD_linear_anlysis_layers_documents_R2_...`

This folder contains supporting materials and analysis results related to the linear RAG experiments.

It may include:

- layer-wise analysis results;
- document-level outputs;
- intermediate experiment results;
- supporting figures or tables.

## Installation

First, clone the repository:

    git clone <repo-url>
    cd ICL_RAG

Install the required Python packages:

    pip install numpy pandas matplotlib scikit-learn torch transformers datasets jupyter

If you use a virtual environment, you can create and activate one first:

    python -m venv icl_rag_env
    source icl_rag_env/bin/activate

Then install the dependencies:

    pip install numpy pandas matplotlib scikit-learn torch transformers datasets jupyter

## Usage

Open the main notebook:

    jupyter notebook ICL_equ_GD.ipynb

Or use JupyterLab:

    jupyter lab

Then run the notebook cells step by step.

The notebook contains the main experimental pipeline and analysis code. Users can modify the parameters, datasets, or model settings to test different cases.

## Requirements

The project is mainly written in Python. Recommended packages include:

- numpy
- pandas
- matplotlib
- scikit-learn
- torch
- transformers
- datasets
- jupyter

## Research Motivation

In standard RAG systems, the retriever selects documents and the generator uses those documents to produce an answer. However, even when the retrieved documents are useful, the model may not use them correctly.

This project studies the generator-side behavior of RAG. In particular, it asks:

- How does the retrieved evidence affect prediction?
- Can ICL be interpreted as an optimization-like process?
- Where does the update happen in a simplified linear RAG setting?
- How does the query interact with retrieved documents?

## Notes

This repository is mainly for research and analysis purposes. The current version focuses on controlled experiments, so users may need to modify the notebook for new datasets, models, or retrieval settings.
