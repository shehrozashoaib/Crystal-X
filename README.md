
# Bridging Atomic and Bond Modalities: Crystal-X Network for Enhanced Bandgap Prediction

Welcome to the official GitHub repository for **Crystal-X** – an advanced graph neural network (GNN) model designed to accurately predict electronic band gaps in materials. This project builds upon the existing Crystal Graph Convolutional Neural Network (CGCNN) architecture by introducing edge-centered convolutions and a novel neighbor feature transformation.

---

# Abstract

Graph neural networks (GNNs) are key tools for predicting material properties at low computational cost by learning crystal structure representations. However, most approaches treat edge attributes as secondary to nodes despite bonds’ critical role in electronic properties. In this study, we introduce Crystal-X, a multimodal GNN architecture that connects atomic and bonding environments through edge-convolutional operations. Current node-centric models fall short in modeling complex bonding environments that determine bandgap values and other material properties. We analyze this limitation by exploring three aspects: performance across space groups, crystal centering types, and bandgap ranges. To address these identified shortcomings, we propose a simple solution: asymmetric edge convolutions with neighbor feature transformations, which directly supplement the missing bond information to GNNs. The resulting architecture achieves a bandgap prediction MAE of 0.26 eV on the Materials Project dataset—improving on CGCNN (0.39 eV), SchNet (0.42 eV), and MEGNet (0.33 eV)—with particular improvements for crystal materials for complex space groups (e.g., monoclinic systems, space groups 3–15), where bond interactions critically influence properties, and non-primitive centering types (e.g., face-centered lattices) with variable bond lengths. We implement granular analysis of the dataset, revealing the impact of data bias on learning and performance in underrepresented band gap classes, and 
identifying high-MAE outliers.

---


## Project Details

### Data Features Creation
- For generating dataset tensors, the algorithm proposed by CGCNN's authors was used. The original file can be viewed in their GitHub repo: https://github.com/txie-93/cgcnn

### Methodology
- **Model Architecture:** Builds upon CGCNN with the following key innovations:
  - **Neighbor Feature Transformation:** Transforms neighboring node features using a shared multi-layer perceptron (MLP).
  - **Asymmetric Edge Function:** Models directional atomic interactions via a learnable edge function.
  - **Edge Feature Aggregation:** Aggregates updated edge features to update node representations.
 
![Crystal-X Model](Architecture.jpg)

- **Training Details:**
  - **Learning Rate:** 1×10⁻³ (using AdamW)
  - **Batch Size:** 256
  - **Convolutional Layers:** 8 (with batch normalization)
  - **Epochs:** 200 (typically converges within 120 epochs)
  - **Hardware:** NVIDIA RTX 4090 GPU, Intel Core i9 CPU

### Key Results

Crystal-X reaches **0.251 eV** MAE on Materials Project band gap.

**Matched-protocol benchmark.** Literature MAEs are reported on each model's own test
split, so they cannot be compared directly. We therefore retrained CGCNN and ALIGNN from
scratch alongside Crystal-X on the *same* MODNet MP_2018.6 data, the *same* 61,640/6,848
split, and evaluated all three on the *same* 6,848 test crystals:

| model | MAE (eV) | RMSE (eV) | R² | wallclock (1×A100) |
|---|---|---|---|---|
| **Crystal-X** | **0.2511** | **0.4983** | **0.9079** | 1.9 h |
| CGCNN | 0.2681 | 0.5105 | 0.9034 | 1.3 h |
| ALIGNN | 0.2632 | 0.5464 | 0.8893 | 11.5 h |

Crystal-X leads on all three metrics and in six of seven crystal systems, at ~1/6 of
ALIGNN's training cost. Full protocol, per-crystal-system and per-band-gap-range
breakdowns, trained weights, run logs and reproduction scripts are in
[`baselines/training/`](baselines/training/) — including a documented caveat that the
validation split is reused as the test split for all three models.

Under this matched protocol the commonly cited single-split figures shift: CGCNN performs
considerably better than its 0.39 eV literature value, and ALIGNN considerably worse than
its 0.22 eV, which was measured on a 4,239-material JARVIS `megnet` split sharing only
~11% of its materials with ours. Comparisons across published splits should be read with
that in mind.

- **Error analysis:** largest gains on metallic compounds (MAE 0.126 vs. CGCNN 0.137 and
  ALIGNN 0.152); wide-gap insulators remain the hardest class for all three models.

### Future Directions
- Predict additional electronic properties.
- Integrate Crystal-X into automated discovery pipelines.
- Expand and diversify the materials dataset to address biases.

---

### Dataset
The Materials Project Dataset 2018 version was downloaded from this URL:
https://figshare.com/articles/dataset/Materials_Project_MP_2018_6_MODData/12834275?file=24364571

The 2022 version was imported using the MP API. The matched benchmark above uses the
MODNet MP_2018.6 version.

---

### Baselines & Benchmark

[`baselines/training/`](baselines/training/) contains everything for the three-model
comparison: training scripts for Crystal-X, CGCNN and ALIGNN, the Slurm jobs used,
per-material test predictions for all three, trained weights, metric summary CSVs, the
per-space-group figure, and the run logs. Start with its README.

### Reproducing & Usage
Our code uses the following libraries:

PyTorch

Pymatgen

Materials Project API (MP_API)

**Running**
To run the notebook and code files, clone this GitHub repo. Make a local directory root_dir in the same folder as the remaining files. Next, save the atom_init.json inside this folder. 

To use the MP 2022 dataset, replace the 'your_api_key' in data.py with your custom API key available upon creating a free account at Materials Project Dataset: https://next-gen.materialsproject.org/

Next, simply run each cell in the notebook.

For MP 2018 dataset, download the MODData file from the URL provided above, along with the MODNet library from pip. Next, replace the dataset filename in the main.ipynb with the data_MP_2018.py, and then simply start running each cell. Make sure to replace the name of the MODData file downloaded from the URL above in the dataset file.

---

### How to Cite
Shehroz Ahmad Shoaib, Burhan SaifAddin. (2025). Bridging Atomic and Bond Modalities: Crystal-X Network for Enhanced Bandgap Prediction [Online]. Available: https://github.com/shehrozashoaib/Crystal-X

---

### Contact
For questions, collaborations, or further information:

Name: Shehroz Ahmad Shoaib
Email: s202353930@kfupm.edu.sa

Name: Burhan SaifAddin
Email: burhan.saifaddin@kfupm.edu.sa

