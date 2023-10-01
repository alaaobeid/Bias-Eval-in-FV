# Bias Evaluation in Face Verification
Face Recognition (FR) technology often exhibits accuracy disparities across different demographic groups, leading to biases in Face Verification (FV) tasks. Assessing and quantifying these biases is essential for ensuring the fairness of FR models. This repository contains the implementation of SOTA bias evaluation metrics in FV, such as, FDR [1], MAPE [2], SER [2], and STD in EER [3].
# Usage
This repository comes with an example jupyter notebook demonstrating how to calculate all the metrics using the calc_fairness_metrics function. The function expects a list of one-dimensional arrays of distances and a list of one-dimensional arrays of labels. The first entry of each of those lists is reserved for the full dataset (global) distance and label arrays respectively.
If you have any further questions regarding the usage please email aladdintahir@gmail.com
# Acknowledgement
This work has received funding from the European Union’s Horizon 2020 research and innovation programme under Marie Sklodowska-Curie Actions (grant agreement number 860630) for the project ‘’NoBIAS - Artificial Intelligence without Bias.
# References
1. Kotwal, K., Marcel, S.: Fairness index measures to evaluate bias in biometric recog-
nition. In: International Conference on Pattern Recognition. pp. 479–493. Springer
(2022)

2. Villalobos, E., Mery, D., Bowyer, K.: Fair face verification by using non-sensitive
soft-biometric attributes. IEEE Access 10, 30168–30179 (2022)

3. Serna, I., Morales, A., Fierrez, J., Obradovich, N.: Sensitive loss: Improving accu-
racy and fairness of face representations with discrimination-aware deep learning.
Artificial Intelligence 305, 103682 (Apr 2022)
