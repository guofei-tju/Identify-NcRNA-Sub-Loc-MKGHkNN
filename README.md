# Identify-NcRNA-Sub-Loc-MKGHkNN

Title:

Identify ncRNA subcellular localization via graph regularized k-local hyperplane distance nearest neighbor model on multi-kernel learning

Abstract:

Non-coding RNAs (ncRNAs) are a type of RNA which are not used to encode protein sequences. Emerging evidence shows that lots of ncRNAs may participate in many biological processes and must be widely involved in many types of cancers. Therefore, understanding their functionality is of great importance. Similar to proteins, various functions of ncRNAs relies on their subcellular localizations. Traditional high-throughput methods in wet-lab to identify subcellular localization is time-consuming and costly. In this paper, we propose a novel computational method based on multi-kernel learning to identify multi-label ncRNA subcellular localizations, via graph regularized $k$-local hyperplane distance nearest neighbor algorithm. First, we construct six types of sequence-based feature descriptors and select important feature vectors. Then, we build a multi-kernel learning model with Hilbert-Schmidt independence criterion (HSIC) to obtain optimal weights for vairous features. Furthermore, we propose the graph regularized $k$-local hyperplane distance nearest neighbor algorithm (GHKNN) as a binary classification model for detecting one kind of non-coding RNA subcellular localization. Finally, we apply One-vs-Rest strategy to decompose multi-label problem of non-coding RNA subcellular localizations. Our method achieves excellent performance on three ncRNA datasets and three human ncRNA datasets. We evaluate our predictor on a novel multi-label benchmark set, and out-performs other outstanding machine learning methods. We expect that this model will be useful for the prediction of subcellular localization and the study of important functional mechanisms of ncRNAs. 

Keywords:
Non-coding RNA, subcellular localization, multi-label classification, multi-kernel learning, k-local hyperplane distance nearest neighbor.

Framework:

![Image text](https://raw.githubusercontent.com/hzhou256/Identify-NcRNA-Sub-Loc-MKGHkNN/main/framework.png)
