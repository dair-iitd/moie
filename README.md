# Multilingual Open Information Extraction

The repository contains the code for the ACL'22 paper: Alignment-Augmented Consistent Translation for Multilingual Open Information Extraction [link](https://aclanthology.org/2022.acl-long.179/)

Please cite the work if it helps you in your research!
```
@inproceedings{kolluru-acl22,
    title = {{A}lignment-{A}ugmented {C}onsistent {T}ranslation for {M}ultilingual {O}pen {I}nformation {E}xtraction},
    author = {Keshav Kolluru and Mohammed Muqeeth and Shubham Mittal and Soumen Chakrabarti and Mausam},
    booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
    address = {Dublin, Ireland},
    month = {May},
    year = {2022},
    publisher = "Association for Computational Linguistics"
}
```

We have a Colab script that includes the commands for running inference on pre-trained models and training the models from scratch.
The Colab link will be visible after submitting this form: [Google Form](https://forms.gle/z8HETfUgbbwAQBFZ6).
It is a very light-weight form that will help us guide future improvements to the system. 
Your cooperation is highly appreciated!

## Using aligner tool

Expects english_labels file to have each line as 'english ext_sentence ||| labels'. The labels must contain ARG1, ARG2, REL, LOC or TIME
The other_lang_extsentence is just the extsentences in the other language

```
python aligner.py --inp1 /path/to/english_labels --inp2 /path/to/other_lang_extsentence --output_file /path/to/other_lang_extraction --alignment_type clp_sentence --lang other_lang --model_name_or_path other_lang_aligner_path
```

