# DoSEA: A Domain-specific Entity-aware Framework for Cross-Domain Named Entity Recognition

This is the source code for COLING 2022 paper: *A Domain-specific Entity-aware Framework for Cross-Domain Named Entity Recognition*.


### Datasets
We use the <a href="https://aclanthology.org/W03-0419/">Conll-2003</a> dataset as source domain, <a href="https://ojs.aaai.org/index.php/AAAI/article/view/17587">CrossNER</a> and <a href="https://doi.org/10.1109/ICASSP.2013.6639301">MIT Movie Review</a> datasets as target domains.

All data can be downloaded from this <a href="https://drive.google.com/file/d/1bNNs-kVZgYd1dEj2knR5GDTr3aX4WNAJ/view?usp=share_link">link</a>.


### Preprocessing
```shell script
python run.py
```


### Citation
If you find this code helpful, please kindly cite the following paper.
```
@inproceedings{tang-etal-2022-dosea,
    title = "{D}o{SEA}: A Domain-specific Entity-aware Framework for Cross-Domain Named Entity Recogition",
    author = "Tang, Minghao  and
      Zhang, Peng  and
      He, Yongquan  and
      Xu, Yongxiu  and
      Chao, Chengpeng  and
      Xu, Hongbo",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.188",
    pages = "2147--2156",
    abstract = "Cross-domain named entity recognition aims to improve performance in a target domain with shared knowledge from a well-studied source domain. The previous sequence-labeling based method focuses on promoting model parameter sharing among domains. However, such a paradigm essentially ignores the domain-specific information and suffers from entity type conflicts. To address these issues, we propose a novel machine reading comprehension based framework, named DoSEA, which can identify domain-specific semantic differences and mitigate the subtype conflicts between domains. Concretely, we introduce an entity existence discrimination task and an entity-aware training setting, to recognize inconsistent entity annotations in the source domain and bring additional reference to better share information across domains. Experiments on six datasets prove the effectiveness of our DoSEA. Our source code can be obtained from https://github.com/mhtang1995/DoSEA.",
}
```
