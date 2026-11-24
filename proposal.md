# Project Proposal

**What Does a Protein Language Model Learn About Epistasis? Studying how pretraining and capacity affects ESM-2 learned representations**

Arya Bhanushali and Aryan Thakkar

## Background and Motivation

Proteins are made up of amino acids, and changing these amino acids can affect how well a protein works. In protein engineering, researchers try to find mutations that improve a protein's function. The problem is that there are so many possible combinations of mutations that testing all of them in a lab isn't realistic. What makes this even harder is that mutations can interact with each other. For example, one mutation might make a protein worse on its own, but when combined with another mutation, it could actually improve the protein. This is called epistasis, and it makes predicting the effects of mutations much more complicated.

Protein language models like ESM-2 have become useful for predicting how mutations affect proteins. These models are pretrained on large amounts of protein sequence data, allowing them to be adapted for tasks like predicting protein fitness. Although these models can perform with high accuracy, we don't know how they have learned key representations in protein structures. For example, two models with the same test error could organize information about mutations in completely different manners.

A study by Rahimimovassagh et al. (2026) explored the nature of prediction models on the GB1 protein with the introduction of ORBIT, which, using the Walsh-Hadamard Transform (WHT), measured interaction information layer by layer using a model's activation info. Their research discovered that adding hidden layers to a regular neural network improved its ability to represent order-three epistasis, meaning three mutations at once. Their study excluded pretrained models, like ESM-2, and focused more on depth analysis. This is the gap we hope to fill. A pretrained ESM-2 and a randomly initialized ESM-2 have the same architecture and parameter count, leaving only whether the model is pretrained or not. The comparison allows us to identify what pretraining contributes and what capacity contributes.

## Problem Formulation and Research Objectives

### Measuring Interaction and Accessibility

For studying fitness landscapes in genetics, the WHT has been used to define interactions between different mutation points by transforming fitness values of sequences into coefficients representing individual mutations. ORBIT applied it to GB1 through the mutation of 4 different positions. Each position represents a point for a mutation to occur or not, giving 16 possible combinations, which is called a cube. The decomposition by WHT rewrites those 16 fitness values from those combinations as interpretable quantities.

As discussed in the ORBIT paper, the interaction representation quantities "does not establish whether order-specific variation is aligned with experimentally measured epistatic structure" (Rahimimovassagh et al. (2026)). It introduces the concept of accessibility, which compares a predicted interaction value with the actual measured value. We predict an interaction value by applying a linear ridge regression on a model's values. We then compute the R-squared error between the linearly probed interaction value and the value obtained using WHT, giving us our accessibility.

### Research Questions

With the ability to measure interaction and accessibility, we define the following research questions given the input/output:

- **Input (raw):** A single GB1 variant, with mutations occurring at positions 39, 40, 41, and 54
- **Input (unit of analysis):** A cube: the 16 variants formed by choosing an alternative amino acid (19 other possible amino acids) at the 4 possible positions. The GB1 data allows for 109,235 complete cubes.
- **Data:** 149361 measured GB1 variants with fitness relative to the original protein (Wu et al., 2016).
- **Intermediate Quantities:** WHT decomposition of each cube, providing the actual interaction coefficient and the representation coefficients at each layer in ESM-2.
- **Output:** Accessibility defined as the R-squared error from the actual interaction coefficient and the predicted interaction coefficient based on the representation coefficients.

1. At an identical architecture and parameter count, does a pretrained protein language model show higher interaction accessibility compared to a randomly initialized ESM-2?
2. Does accessibility scale for higher-order epistasis with higher model capacity?
3. Where does accessibility begin to appear in models?

## Related Work

As mentioned earlier, ORBIT (Rahimimovassagh et al., 2026) is the predecessor. It examines where interaction information is represented in a model and whether that can be linearly extracted. Another paper was InterPLM (Simon et al., 2024), which looked inside ESM-2 using sparse autoencoders. The authors found features related to different protein structures and functions. This showed that some of the information learned by protein language models can be interpreted. Other papers, including work by Wu et al. (2016) and Wittmann et al. (2021), have shown that machine learning can help predict protein fitness and identify useful mutations without testing every possible combination in a lab.

Another paper by Ahlmann-Eltze et al. (2025) found that complex deep learning models don't always outperform simple baselines when predicting the effects of genetic perturbations. This made us wonder if something similar could happen with protein fitness prediction. If pretraining provides less utility than predicted, random initialization should show similar accessibility.

## Proposed Methodology

For this project, our research questions are answered by running GB1 variants through a model and probing the activations.

The model we will be using is the facebook/esm2 model family from Hugging Face. In terms of pretraining, one set of models will be trained on the UniRef50 dataset, consisting of millions of protein sequences. The other set of models will be randomly initialized. The capacities of the models being tested are 8M, 35M, and 150M parameters.

For the procedure, we will build all 109,235 possible cubes, matching the ORBIT paper. From there, we will calculate the ground truth using WHT. Then, we will train each model by LoRA adapting them on the training split with a regression head. We then pass the GB1 variants and cache activations at every layer. We probe each layer using linear ridge regression to obtain the predicted interaction coefficient. Finally, we compare accessibility curves across different orders based on pretrained vs. random initialization and varying capacities.

For training-test splits, we will use the official FLIP GB1 2-vs-rest split to match the ORBIT paper, dividing the dataset to train on variants with at most 2 mutations and then testing the models on three- and four-way mutations.

The data we will be using is the GB1 four-site fitness landscape from Wu et al. (2016). The authors mutated the protein at positions 39, 40, 41, and 54, measuring fitness for each variant. These four positions were chosen as mutation points because the authors determined that these positions result in the most positive pairwise interactions in GB1.

## Evaluation Plan and Expected Outcomes

### Experiments

- **Verify:** We begin by verifying a correct approach for calculating the actual and predicted coefficients. We use a simple additive model and see if it is possible to reproduce ORBIT's published results.
- **Pretrain vs. Random:** We compare a pre-trained 150M ESM-2 and randomly initialized ESM-2, both trained identically from there.
- **Capacity:** We compare pretrained ESM-2 at sizes 8M, 35M, and 150M, and determine how size impacts interaction.
- **Layer-represented Interaction:** Using the cached activations from previous experiments, we probe different layers to determine where interaction and accessibility are built in a model.

### Metrics

- **Accessibility:** The R-squared error between the actual interaction value and the predicted value from the ridge probe that predicted the interaction value from layer activations.
- **Spearman rho on the test split:** The rank correlation to determine ordinary prediction quality.
- **R-squared error on fitness:** To determine the prediction quality based on error difference.

### Baselines

We will be using the additive model from the ORBIT paper as a baseline to anchor our results. The additive model predicts fitness as the sum of singular mutation effects.

### Expected Outcomes

For this project, we expect that pretraining will result in higher accessibility due to its ability to help ESM better understand interactions. We also expect that higher capacity will result in higher accessibility significantly more than just adding pretraining. Based on the Ahlmann-Eltze et al. (2025) paper, we believe that the model's results are largely dependent on its learning capabilities rather than initial weights.

## References

1. Rahimimovassagh, M., Garibay, I., Yousefi, N. (2026). Beyond tokens: probing higher-order epistasis in learned protein representations. https://arxiv.org/abs/2608.24953
2. Wu, N.C., Dai, L., Olson, C.A., Lloyd-Smith, J.O., Sun, R. (2016). Adaptation in protein fitness landscapes is facilitated by indirect paths. eLife 5:e16965. https://doi.org/10.7554/eLife.16965
3. Dallago, C., Mou, J., Johnston, K.E., Wittmann, B.J., Bhattacharya, N., Goldman, S., Madani, A., Yang, K.K. (2021). FLIP: benchmark tasks in fitness landscape inference for proteins. NeurIPS Datasets and Benchmarks Track. https://benchmark.protein.properties/
4. Simon, E., Zou, J. (2024). InterPLM: discovering interpretable features in protein language models via sparse autoencoders. https://arxiv.org/abs/2412.12101
5. Ahlmann-Eltze, C., Huber, W., Anders, S. (2025). Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines. Nature Methods 22(8):1657-1661. https://pmc.ncbi.nlm.nih.gov/articles/PMC12328236/
6. Wittmann, B.J., Yue, Y., Arnold, F.H. (2021). Informed training set design enables efficient machine learning-assisted directed protein evolution. Cell Systems 12(11):1026-1045.
