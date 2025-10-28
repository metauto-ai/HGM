<h1 align="center">
   Huxley-Gödel Machine: Human-Level Coding Agent Development by an Approximation of the Optimal Self-Improving Machine
</h1>

<p align="center">
  <img src="./misc/hgm.png" width="40%" height="auto" />
</p> 

Repository for **[Huxley-Gödel Machine](https://arxiv.org/abs/2510.21614)** (HGM), an approximation of the Gödel Machine: the theoretical optimal self-improving machine. HGM makes this concept practical with coding agents that iteratively rewrite themselves, using estimates of the promise of entire subtrees (clades) to decide which self-modifications to expand.

## Setup
```bash
# API keys, add to ~/.bashrc
export OPENAI_API_KEY='...'
```

```bash
# Verify that Docker is properly configured in your environment.
docker run hello-world
 
# If a permission error occurs, add the user to the Docker group
sudo usermod -aG docker $USER
newgrp docker
```

```bash
# Install dependencies
conda create -n hgm
pip install -r requirements.txt
```

```bash
# Clone SWE-bench
cd swe_bench
git clone https://github.com/princeton-nlp/SWE-bench.git
cd SWE-bench
git checkout dc4c087c2b9e4cefebf2e3d201d27e36
pip install -e .
cd ../../

# Prepare Polyglot
# Make sure git is properly configured in your environment with username and email
python -m polyglot.prepare_polyglot_dataset
```

## Running the HGM
```bash
./run.sh
```

## Safety Consideration
> [!WARNING]  
> This repository involves executing untrusted, model-generated code. We strongly advise users to be aware of the associated safety risks. While it is highly unlikely that such code will perform overtly malicious actions under our current settings and with the models we use, it may still behave destructively due to limitations in model capability or alignment. By using this repository, you acknowledge and accept these risks.

## Acknowledgement

The code in this repository is built upon the code from the [Darwin-Gödel Machine](https://github.com/jennyzzt/dgm/tree/main). We thank the authors for making their code publicly available. The evaluation framework implementations are based on the [SWE-bench](https://github.com/swe-bench/SWE-bench) and [polyglot-benchmark](https://github.com/Aider-AI/polyglot-benchmark) repositories.

## Reference

```bash
@misc{wang2025huxleygodelmachinehumanlevelcoding,
      title={Huxley-G\"odel Machine: Human-Level Coding Agent Development by an Approximation of the Optimal Self-Improving Machine}, 
      author={Wenyi Wang and Piotr Piękos and Li Nanbo and Firas Laakom and Yimeng Chen and Mateusz Ostaszewski and Mingchen Zhuge and Jürgen Schmidhuber},
      year={2025},
      eprint={2510.21614},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.21614}, 
}
```
