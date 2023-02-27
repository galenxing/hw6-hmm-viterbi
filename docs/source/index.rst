.. hw6-hmm documentation master file, created by
   sphinx-quickstart on Sat Feb 11 16:27:34 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Lab 6: Inferring CRE Selection Strategies from Chromatin Regulatory State Observations using a Hidden Markov Model and the Viterbi Algorithm
============================================================================================================================================

The aim of hw6 is to implement the Viterbi algorithm, a dynamic program that is a common decoder for Hidden Markov Models (HMMs). The lab is structured by training objective, project deliverables, and experimental deliverables:

**Training Objective**: Learn how to design reusable Python packages with automated code documentation and develop testable (user case) hypotheses using the Viterbi algorithm to decode the best path of hidden states for a sequence of observations.

**Project Deliverable**: Produce a simple report for functional characterization inferred from a binary regulatory observation state pattern across cardiac developmental timepoints.

**Experimental Deliverable**: Construct a positive control library for massively parallel reporter assays (MPRAs) and CRISPRi/a experiments in primitive and progenitor cardiomyocytes (i.e., cardiogenomics).

Key Words
==========
Chromatin; histones; nucleosomes; genomic element; accessible chromatin; chromatin states; genomic annotation; candidate cis-regulatory element (cCRE); Hidden Markov Model (HMM); ENCODE; ChromHMM; cardio-genomics; congenital heart disease(CHD); TBX5


Functional Characterization Report
===================================

Please evaluate the project deliverable and briefly answer the following speculative question, with an eye to the project's limitations as related to the theory, model design, experimental data (i.e., biology and technology). We recommend answers between 2-6 sentences. It is OK if you are not familiar already with this biological user case; you can receive full points for your best-effort answer.

1. Speculate how the progenitor cardiomyocyte Hidden Markov Model and primitive cardiomyocyte regulatory observations and inferred hidden states might change if the model design's sliding window (default set to 60 kilobases) were to increase or decrease?

If you increase the sliding window you might be adding too much extraneous information. Thus it would introduce more noise and could lead to less accurate results. Similarly if you descreased the sliding window too much you might not have enough data to make proper inferences. 

2. How would you recommend integrating additional genomics data (i.e., histone and transcription factor ChIP-seq data) to update or revise the progenitor cardiomyocyte Hidden Markov Model? In your updated/revised model, how would you define the observation and hidden states, and the prior, transition, and emission probabilities? Using the updated/revised design, what new testable hypotheses would you be able to evaluate and/or disprove?

If you were to integrate more data, you would need to recalculate the prior, transition, and emission probabilities. I would do this by running a model on the before and after integration and see if the addition of data improved the hidden state prediction. You should also use your priors about TF biology in order to determine whether or use more or less data.

3. Following functional characterization (i.e., MPRA or CRISPRi/a) of progenitor and primitive cardiomyocytes, consider all possible scenarios for recommending how to update or revise our genomic annotation for *cis*-candidate regulatory elements (cCREs) and candidate regulatory elements (CREs)?

You should have a probabilistic bar for what is and isnt a cCRE or a CRE. Then following new data and running optimizations to your model (e.g. playing around with parameters usch as the window sizes) you can then determine what regions to classify as cCRE and CRE based on the stability of the predictions. These should then be compared with the new functional characterization and if it matches, then you can update your genomic annotation. 


Models Package 
======================
.. toctree::
   :maxdepth: 2
   
   modules
