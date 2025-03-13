# Crystal-X: A GNN-Based Approach for Accurate Band Gap Prediction

![Crystal-X Model](Architecture.jpg)

# Crystal-X: A Novel GNN-Based Approach for Accurate Band Gap Prediction

Welcome to the official GitHub repository for **Crystal-X** – an advanced graph neural network (GNN) model designed to accurately predict electronic band gaps in materials. This project builds upon the existing Crystal Graph Convolutional Neural Network (CGCNN) architecture by introducing edge-centered convolutions and a novel neighbor feature transformation.

---

## Overview

The accurate prediction of band gaps is essential for the discovery of new semiconductors used in solar cells, wide bandgap materials, and electronic devices. **Crystal-X** addresses the computational bottlenecks of traditional methods such as density functional theory (DFT) by leveraging GPU acceleration and dynamic edge feature learning. Our model demonstrates a significant reduction in mean absolute error (MAE), achieving 0.25 eV compared to 0.39 eV with CGCNN and 0.33 eV with MEGNet.

---


## Project Details


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
- **Performance:** Achieves a MAE of 0.25 eV on band gap prediction using the Materials Project dataset.
- **Comparative Analysis:** Outperforms CGCNN (0.39 eV) and MEGNet (0.33 eV), with performance approaching ALIGNN (0.22 eV) but at a lower computational cost.
- **Error Analysis:** Demonstrates significant improvements in metallic compounds while highlighting challenges in semiconductor and insulator predictions.

### Future Directions
- Predict additional electronic properties.
- Integrate Crystal-X into automated discovery pipelines.
- Expand and diversify the materials dataset to address biases.

---

### How to Cite
Shehroz Ahmad Shoaib, Burhan SaifAddin. (Year). Crystal-X: A Novel GNN-Based Approach for Accurate Band Gap Prediction. [Online]. Available: https://github.com/yourusername/Crystal-X

---

### Contact
For questions, collaborations, or further information:

Name: Shehroz Ahmad Shoaib
Email: s202353930@kfupm.edu.sa

