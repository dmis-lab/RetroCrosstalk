# Retrosynthetic Crosstalk between Single-step Reaction and Multi-step Planning

## Description
This project investigates the interplay between single-step retrosynthesis models and multi-step planning algorithms to enhance route generation in drug discovery. We evaluate various model-algorithm combinations across diverse datasets, measuring both route completion (solvability) and practical feasibility. Our results show that optimizing solely for solvability does not guarantee feasible synthetic routes, offering crucial insights into real-world computational retrosynthesis.


## Setup
### 1. Default and ReactionT5 Environment
```bash
# Clone the repository
git clone https://github.com/dmis-lab/RetroCrosstalk.git
cd RetroCrosstalk

# Create and activate a virtual environment (optional but recommended)
conda create -n default_rxnt5 python=3.9
conda activate default_rxnt5

# Install dependencies
pip install -r default_rxnt5_requirements.txt
```

### 2. AZF Environment
```bash
# Create and activate a virtual environment (optional but recommended)
conda create -n azf python=3.9
conda activate azf

# Install dependencies
pip install -r azf_requirements.txt
```

### 3. LocalRetro and Chemformer Environment
```bash
# Create and activate a virtual environment (optional but recommended)
conda create -n local_chem python=3.9
conda activate local_chem

# Install dependencies
pip install -r local_chem_requirements.txt
pip install dgl==1.1.3+cu118 -f https://data.dgl.ai/wheels/cu118/repo.html
python -m pip install git+https://github.com/MolecularAI/pysmilesutils.git
```

### 4. Metric Environment
```bash
# Create and activate a virtual environment (optional but recommended)
conda create -n metric python=3.9
conda activate metric

# Install dependencies
pip install -r metric_requirements.txt
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
```

### 5. Visualization  Environment
```bash
# Create and activate a virtual environment (optional but recommended)
conda create -n viz python=3.9
conda activate viz

# Install dependencies
pip install -r viz_requirements.txt
conda install -c conda-forge pycairo
pip install cairosvg
```

### 6. Data Preparation
```bash
cd Data
gzip -d eMolecules.txt.bz2.gz
bzip2 -d eMolecules.txt.bz2
tar -zcvf PaRoues.tar.gz
```

### 7. Additional Files Setup Instructions
To use this repository, you need to download and set up additional files. Please follow these instructions:

Download the required files from the following Google Drive link:

[Google Drive Download Link](https://drive.google.com/drive/folders/16iBf9VYIAbpFXND8KnXWfKbZ6M4NbSAS?usp=sharing)

Save the downloaded files to the RetroCrosstalk folder.

Unzip the downloaded file within the RetroCrosstalk folder:
```bash
cd RetroCrosstalk
unzip downloaded_filename.zip
```
After extracting the files, you can proceed with the project.


## Usage
Explain how to use your project with some examples.

```bash
# Run inference using MEEA* with default settings
cd meea
sh run_meea.sh
```
```bash
# Evaluate results for a specific model and dataset
python score.py -ss [ss_model] -d [dataset]
```
```bash
# Visualize the retrosynthetic tree for a given route
python visualize_tree.py
```


## License
MIT License – See the [LICENSE](LICENSE) file for details.

## Contact
Junseok Choe - [juns94@korea.ac.kr](mailto:juns94@korea.ac.kr)

Project Link: [RetroCrosstalk GitHub](https://github.com/dmis-lab/RetroCrosstalk)

