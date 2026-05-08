# Hierarchical Histogram Threshold Segmentation (HHTS)
### Superpixel Segmentation for Satellite and Natural Imagery
[**Live Demo**](https://huggingface.co/spaces/AbdallahAdel/HTTS_implementation) | [**Project Notebook**](https://www.kaggle.com/code/abdallahadelabdallah/hhts-implementation)


## Abstract
Superpixel segmentation is a fundamental preprocessing step in computer vision, yet traditional methods like SLIC often smooth over critical irregular boundaries. This project implements Hierarchical Histogram Threshold Segmentation (HHTS), an iterative top-down approach that utilizes 1D Laplace filtering and Cauchy-weighted thresholding to identify precise intensity clusters. To ensure the fidelity of our implementation, we validated its performance on the BSDS500 benchmark, achieving results consistent with the original paper. We also demonstrate the approach's application on satellite imagery using the DeepGlobe Land Cover Classification dataset.


## HHTS Implementation
Our project focuses on a robust implementation of the HHTS algorithm, a "divide and conquer" strategy for image partition. Unlike standard clustering methods that force pixels into a grid, HHTS treats the image as a global collection of evolving segments.

### Key Logic:
*   **Global Priority Selection:** The algorithm maintains a priority queue of segments, always choosing the largest and most "complex" (highest variance) segment to split next.
*   **Intelligent Thresholding:** By analyzing the color distribution (histogram) of a specific segment, the algorithm identifies the most significant boundaries between object classes.
*   **Spatial Connectivity:** After every split, the algorithm ensures that resulting segments are physically connected. It includes a cleanup phase where "tiny" fragments (below 64 pixels) are merged into larger neighboring regions to maintain meaningful segmentation.
*   **Multi-Channel Support:** To ensure no detail is missed, the implementation calculates variance across 9 different color channels, including RGB, HSV, and LAB color spaces.


## Using HHTS for Remote-Sensing Superpixel Segmentation
Remote sensing data (Satellite imagery) contains thin, complex features such as roads, rivers, and varying forest density. Traditional "compact" superpixels often merge these thin features into the background. HHTS excels here because its boundaries are driven by color clusters rather than spatial grids.

![](https://github.com/abanoub-samy-farhan/hhts_implementation/blob/main/images/img1.jpeg)

## Experiments and Results
Our experimental evaluation is divided into two phases: a real-world **application** on satellite imagery and a quantitative **validation** against the standard natural image benchmark used in the original paper.

### 4.1 Application: Remote Sensing (DeepGlobe Dataset)
To evaluate the performance of HHTS on high-resolution satellite data, we conducted a head-to-head comparison with the SLIC baseline on samples from the **DeepGlobe Land Cover Classification Dataset**. This test assesses the algorithm's ability to preserve intricate land-cover boundaries.

| Image | Method | num_segments | runtime_sec | ICV (↓) | EV (↑) | CO (↑) | edge_align | BR (↑) | UE (↓) | ASA (↑) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 100877_sat.jpg | **HHTS** | 1000 | 15.351 | 8.1593 | 0.8017 | 0.2541 | 0.3420 | 0.8768 | 0.9804 | 0.0196 |
| 100877_sat.jpg | SLIC | 978 | 0.329 | 8.8492 | 0.7536 | 0.5897 | 0.3852 | 0.7168 | 0.9806 | 0.0194 |
| 103215_sat.jpg | **HHTS** | 1003 | 12.716 | 13.5552 | 0.8156 | 0.2744 | 0.7391 | 0.8692 | 0.9917 | 0.0083 |
| 103215_sat.jpg | SLIC | 939 | 0.361 | 15.1002 | 0.7581 | 0.4406 | 0.7644 | 0.7772 | 0.9917 | 0.0083 |
| 103742_sat.jpg | **HHTS** | 1000 | 17.909 | 10.0575 | 0.8424 | 0.2585 | 0.4662 | 0.8483 | 0.9850 | 0.0150 |
| 103742_sat.jpg | SLIC | 969 | 0.325 | 10.6806 | 0.7598 | 0.6044 | 0.4979 | 0.7147 | 0.9851 | 0.0149 |
| 110224_sat.jpg | **HHTS** | 1002 | 14.891 | 5.2393 | 0.8407 | 0.2596 | 0.1496 | 0.8624 | 0.9724 | 0.0276 |
| 110224_sat.jpg | SLIC | 1001 | 0.319 | 5.6347 | 0.7860 | 0.7300 | 0.1757 | 0.6665 | 0.9735 | 0.0265 |
| 112946_sat.jpg | **HHTS** | 1001 | 16.053 | 9.2234 | 0.8918 | 0.2700 | 0.4369 | 0.8559 | 0.9812 | 0.0188 |
| 112946_sat.jpg | SLIC | 925 | 0.325 | 10.2774 | 0.8349 | 0.5864 | 0.4904 | 0.7044 | 0.9820 | 0.0180 |

**Key Findings:**
*   **Boundary Precision:** HHTS achieves significantly higher **Boundary Recall (BR)** and **Explained Variation (EV)**, indicating it captures physical terrain boundaries with much higher fidelity than the grid-constrained SLIC.
*   **Metric Trade-offs:** While SLIC is optimized for speed and produces more **compact (CO)** segments, HHTS prioritizes internal homogeneity (**ICV**), resulting in superpixels that better represent distinct land-cover classes.

### 4.2 Validation: Berkeley Segmentation Dataset 500 (BSDS500)
To validate our implementation, we conducted a large-scale experiment on the **BSDS500 test set** (200 images). This experiment was specifically designed to verify that our code reproduces performance curves reported in the original HHTS research paper.

### Performance Curves
The following graphs illustrate the performance of HHTS versus SLIC across varying superpixel counts.

![BSDS500 Benchmark Graphs](https://github.com/abanoub-samy-farhan/hhts_implementation/blob/main/images/BSDS500_results.png)
#### Validation Summary:
*   **Reproduction of Trends:** Our implementation successfully mirrors the findings of the original paper, specifically the steep improvement in **Boundary Recall (BR)** and **Achievable Segmentation Accuracy (ASA)** compared to SLIC.
*   **Error Minimization:** The **Undersegmentation Error (UE)** is consistently lower for HHTS, confirming that our hierarchical splitting and auto-termination logic effectively prevent segments from "bleeding" across multiple object boundaries

## Conclusion and Future Work
Our implementation proves that a hierarchical, histogram-based approach is superior for tasks where object outlines are more important than uniform segment sizes. HHTS is a powerful tool for satellite imagery analysis, providing a high-detail map of the Earth's surface that traditional methods cannot match.

**Future Work:**
*   Moving the histogram calculations to GPU to enable real-time processing of high-resolution satellite tiles.
*   Integrating deep learning features into the priority queue to prioritize specific object classes (e.g., "always split roads first").

## References
1.  [Riedel et al., "Hierarchical Histogram Threshold Segmentation – Auto-terminating High-detail Oversegmentation," 2023.](https://openaccess.thecvf.com/content/CVPR2024/papers/Chang_Hierarchical_Histogram_Threshold_Segmentation_-_Auto-terminating_High-detail_Oversegmentation_CVPR_2024_paper.pdf)
2.  [BSDS500 Dataset](https://www.kaggle.com/datasets/balraj98/berkeley-segmentation-dataset-500-bsds500)
3.  [DeepGlobe Land Cover Classification Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset)
