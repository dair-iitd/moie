# Multilingual Open Information Extraction

The repository contains the code for the ACL'22 paper: Alignment-Augmented Consistent Translation for Multilingual Open Information Extraction [paper link](https://aclanthology.org/2022.acl-long.179/)

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

## Training and Inference with the Gen2OIE model 

We have a Colab script that includes the commands for running inference on trained models that replicates the results reported in the paper and training the models from scratch.
The Colab link will be visible after submitting this form: [Google Form](https://forms.gle/z8HETfUgbbwAQBFZ6).
It is a very light-weight form that will help us guide future improvements to the system. 
Your cooperation is highly appreciated!

The predictions of the model on the standard CaRB set are provided in the outputs folder in the repository.

## Generating the AACTrans data

To generate AACTrans data from English to say Spanish, the below steps have to be performed. You can use any translation
system of your choice. In our experiments, we used the fairseq [link](https://github.com/facebookresearch/fairseq/).

Step 1: Train a translation system from English to Spanish using a parallel corpus. Let us call this the Trans_En_Es system.

Step 2: Train an alignment system between English and Spanish using parallel sentences used in previous step. Let us call this the Align_En_Es system.
```
python aligner_train.py  --output_dir=/path/to/alignment_model --model_name_or_path=bert-base-multilingual-cased --extraction 'softmax' --do_train --train_tlm --train_so --train_data_file=/path/to/train_data_file --per_gpu_train_batch_size 2   --gradient_accumulation_steps 4 --num_train_epochs 5 --learning_rate 2e-5 --save_steps 5000 --max_steps 40000 --do_eval --eval_data_file=/path/to/eval_data_file --train_mlm --train_tlm_full --train_psi
```

Step 3: Align and concatenate the Spanish output text with English input text in the parallel corpus using Align_En_Es.
```
python aligner.py --inp1 /path/to/english_sentences  --inp2 /path/to/spanish_sentences --output_file /path/to/cons_trans_input --alignment_type translation_sentence --lang lang_id --model_name_or_path /path/to/alignment_model 
```

Step 4: Train a translation system that uses aligned text as input and the original Spanish text itself as the output. Let us call this the AACTrans_En_Es system.

Step 5: Translate the English Open IE corpus using Trans_En_Es.

Step 6: Now align and concatenate the English and Spanish sentences, ext_sentences using Align_En_Es.
```
python aligner.py --inp1 /path/to/openie_sentences --inp2 /path/to/openie_translations --output_file /path/to/cons_trans_input --alignment_type translation_sentence --lang lang_id --model_name_or_path /path/to/alignment_model 
```

Step 7: Re-translate the aligned and concatenated inputs using AACTrans_En_Es.

Step 8: Project the labels from English extraction to the final translated Spanish ext_sentence using Align_En_Es.
```
python aligner.py --inp1 /path/to/openie_english_labels  --inp2 /path/to/openie_translations --output_file /path/to/openie_translated_labels --alignment_type clp_sentence  --lang lang_id --model_name_or_path /path/to/alignment_model 
```

## Using the label projection tool 

If you want to project labels from English extraction to a translated extsentence, this command will help!

Expects english_labels file to have each line as 'english ext_sentence ||| labels'. The labels must contain ARG1, ARG2, REL, LOC or TIME tags.

The translated_extsentences is the translation of English extsentence to the other language.

```
python aligner.py --inp1 /path/to/english_labels --inp2 /path/to/translated_extsentences --output_file /path/to/translated_labels --alignment_type clp_sentence --lang other_lang --model_name_or_path other_lang_aligner_path
```

