# PhaLSP-DPs

Code and Datasets for "Semantic fusion of dual perspectives on genomic sequences and quorum sensing for bacteriophage lifestyle prediction"

### Developers

Zhen Xiao (xiaozh@mails.ccnu.edu.cn) and Xingpeng Jiang (xpjiang@mail.ccnu.edu.cn) from School of Computer Science, Central China Normal University.

### Datasets

- data/dataset_1865.csv is is a dataset containing 1865 phage names, Accession number, lifestyle and their host information.
- data/data1865_label.csv is is a dataset containing the sample names and lifestyle labels of 1865 phages.
- data/QS_data/1865phage_qs_feature.csv is a dataset representing the QS-based features.
- data/QS_data/feature_count.csv is the detailed ranking results of the quorum sensing molecules across all bacteria.
- data/QS_data/raw_QS is a set of files with three raw host quorum sensing datasets.
- data/DNA_features is a set of files with features encoded by SNPs-based local feature learning module.
- data/data_contigs is a set of files with DNA fragments of diverse lengths.
- data/case is a set of files with information of crAssphages and biologically experiment-Verified Phages.
- bert-base-uncased is a folder contains the pre-trained weights, configuration files, and tokenizer vocabulary for the DNABERT model.

### Environment Requirement
The code has been tested running under Python 3.7. The required packages are as follows:

* numpy == 1.21.5
* pandas == 1.3.5
* torch == 1.4.0+cpu

We utilized DNABERT model to train the global representations of phage sequences. You can follow the instructions provided at [https://github.com/jerryji1993/DNABERT](https://github.com/jerryji1993/DNABERT) to install DNABERT and set up the DNABERT Conda environment.

### Usage
###  Prepare the dataset 

```
git clone https://github.com//PhaLSP-DPs    
cd PhaLSP-DPs/code
```
```
python position_feature.py`    ##Learn the local features from phage DNA sequences
python contigs_processing.py   ##Obtain short contigs of different lengths
```

Users can use their own data to train prediction models. 

1. Download all phage nucleotide sequences from [NCBI batchentrez](https://www.ncbi.nlm.nih.gov/sites/batchentrez?), named **phage1865.fasta** into the *data* folder.
2. Run `python position_feature.py` to learn the local features derived from phage DNA sequences. The relative position matrix of nucleotide fragments extracted at different intervals can be obtained by adjusting the parameter *d*. Save these features into the *data/DNA_features/* folder.
3. Run `python DNABERT_representation.py` to learn the global features from phage DNA sequences. 
4. Use `contigs_processing.py` to artificially divide phage DNA sequences into short contigs of different lengths. Obtain short contigs of different lengths by adjusting the parameter *fragment_length*.
5. To obtain QS-related feature representations of phages for your own dataset, you need phage-host interactions data. The relevant feature representations can be generated using the scripts provided in the *code/QS_processing/* directory.


###  Predicting phage lifestyle

Users can run `python model.py` to execute PhaLSP-DPs on complete phage genome sequence data. 

**Note** If you want to obtain results on short contigs, you need to first use `contigs_processing.py` to generate short contigs of various lengths. Then, the processed results should be generated using `position_feature.py` and `DNABERT_representation.py`, and organized into their respective folders. Finally, you need to update the file paths and execute `python model.py`.

### Case study

We provide the processing data of crAssphages in the case study, and you can directly run `python case.py` to predict the results. Other case studies can also be predicted through the above process.


### Contact

Please feel free to contact us if you need any help.
