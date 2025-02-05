This repository contains a Python implementation for predicting the latitude and longitude of test images based on a reference dataset of labeled images. The approach leverages deep learning models to extract image embeddings and uses cosine similarity to determine the closest matches.

***Overview***

The project includes two implementations:

**VGG16-Based Prediction:** Uses the pre-trained VGG16 model to extract deep features from images.

**CLIP-Based Prediction:** Uses OpenAI's CLIP model to extract semantic embeddings, allowing for better generalization.

**Hybrid Approach:** Enhances the CLIP-based approach with:

A similarity threshold to filter unreliable matches.

A weighted average of latitude and longitude, considering similarity scores.

***Explanation of Methods***

**1. VGG16-Based Approach**

Uses a pre-trained VGG16 model to extract CNN-based deep features.

Computes cosine similarity between test and training images.

Uses a fixed threshold (0.8) to filter unreliable matches.

Averages latitude/longitude of similar images.

**2. CLIP-Based Approach**

Uses OpenAI's CLIP model (ViT-B/32) for embedding extraction.

More semantic understanding, generalizing better than CNN-based features.

Computes cosine similarity, but without a threshold.

Predicts lat/lon using an unweighted average of top-4 matches.

**3. Hybrid CLIP-Based Approach (Improved)**

Adds a similarity threshold: Filters out low-confidence matches.

Uses weighted averaging: More similar images contribute more to lat/lon estimation.

Provides better accuracy in cases where CLIP embeddings perform well but need fine-tuning.
