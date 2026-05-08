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

## Experiment: Berkeley Segmentation Dataset 500 (BSDS500)
To quantitatively validate our implementation, we ran a large-scale experiment on the **BSDS500** test set. We compared HHTS against the industry-standard **SLIC** baseline across the entire 200-image test set.
### Performance Curves
The following graphs illustrate the clear advantage of HHTS in boundary adherence. While SLIC produces "prettier" square segments (higher Compactness), HHTS provides significantly more accurate object boundaries (higher Boundary Recall).
![BSDS500 Benchmark Graphs](https://github.com/abanoub-samy-farhan/hhts_implementation/blob/main/images/BSDS500_results.png)

## Conclusion and Future Work
Our implementation proves that a hierarchical, histogram-based approach is superior for tasks where object outlines are more important than uniform segment sizes. HHTS is a powerful tool for satellite imagery analysis, providing a high-detail map of the Earth's surface that traditional methods cannot match.

**Future Work:**
*   Moving the histogram calculations to GPU to enable real-time processing of high-resolution satellite tiles.
*   Integrating deep learning features into the priority queue to prioritize specific object classes (e.g., "always split roads first").

## References
1.  [Riedel et al., "Hierarchical Histogram Threshold Segmentation – Auto-terminating High-detail Oversegmentation," 2023.](https://openaccess.thecvf.com/content/CVPR2024/papers/Chang_Hierarchical_Histogram_Threshold_Segmentation_-_Auto-terminating_High-detail_Oversegmentation_CVPR_2024_paper.pdf)
2.  [BSDS500 Dataset](https://www.kaggle.com/datasets/balraj98/berkeley-segmentation-dataset-500-bsds500)
3.  [DeepGlobe Land Cover Classification Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset)
