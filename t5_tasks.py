import t5
import os
import functools
import tensorflow as tf
#from t5.data import sentencepiece_vocabulary
import t5.data
from t5.evaluation import metrics

DATA_DIR = "gs://moie_bucket/data/"

DEFAULT_SPM_PATH = "gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model"

#DEFAULT_VOCAB = sentencepiece_vocabulary.SentencePieceVocabulary(
#    DEFAULT_SPM_PATH)
DEFAULT_VOCAB = t5.data.SentencePieceVocabulary(DEFAULT_SPM_PATH)

DEFAULT_OUTPUT_FEATURES = {
    "inputs": t5.data.Feature(
        vocabulary=DEFAULT_VOCAB, add_eos=True, required=False),
    "targets": t5.data.Feature(
        vocabulary=DEFAULT_VOCAB, add_eos=True)
}


def get_downloaded_data_path(data_dir1, split, extension):
    return os.path.join(data_dir1, split + extension)


def preprocess(
        dataset,
        prefix='',  # not used
        sample_answer=False,  # not used
):
    def data_map(ex):
        """Map Natural Questions example to text-to-text example."""
        input = ex['input']
        target = ex['target']

        return {'inputs': input, 'targets': target, 'answers': target}

    dataset = dataset.map(
        data_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset.filter(lambda ex: tf.strings.length(ex['targets']) > 0)


def dataset_fn(split, shuffle_files=False, dataset=""):
    # Load lines from the text file as examples.
    ds = tf.data.TextLineDataset(get_downloaded_data_path(DATA_DIR + dataset, split, ".tsv"))
    print(" >>>> about to read tsv . . . ")
    ds = ds.map(
        functools.partial(tf.io.decode_csv, record_defaults=["", ""], use_quote_delim=False, field_delim="\t"),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda *ex: dict(zip(["input", "target"], ex)))
    return ds


def postprocessor(answer, example=None, is_target=False):
    """Returns answer, or all answers if the full example is provided."""
    if is_target:
        return example["answers"]
    return answer


# t5.data.TaskRegistry.add(
#     f"parsiglue_readingcomprehension",
#     # Supply a function which returns a tf.data.Dataset.
#     dataset_fn=functools.partial(dataset_fn, dataset="parsiglue_readingcomprehension"),
#     splits=["train", "dev", "eval"],
#     # Supply a function which preprocesses text from the tf.data.Dataset.
#     text_preprocessor=preprocess,
#     # Lowercase targets before computing metrics.
#     postprocess_fn=postprocessor,
#     output_features=DEFAULT_OUTPUT_FEATURES,
#     metric_fns=[metrics.squad],
# )


# datasets = ['pt/mt5/vanilla', 'pt/mt5/consistent', 'pt/genoie/clp', 'pt/genoie/translate_clp', 'pt/genoie/ctranslate_clp', 'pt/gen2oie_s1/clp', 'pt/gen2oie_s1/translate_clp', 'pt/gen2oie_s1/ctranslate_clp', 'pt/gen2oie_s2/clp', 'pt/gen2oie_s2/translate_clp', 'pt/gen2oie_s2/ctranslate_clp', 'pt/genoie/rerank', 'pt/gen2oie_s1/ctranslate_clp_wopos', 'pt/gen2oie_s1/translate_clp_wopos', 'pt/gen2oie_s1/clp_wopos', 'pt/gen2oie_s2/ctranslate_clp_wopos', 'pt/gen2oie_s2/translate_clp_wopos', 'pt/gen2oie_s2/clp_wopos']
# datasets += ['te/mt5/vanilla', 'te/mt5/consistent', 'te/genoie/clp', 'te/genoie/translate_clp', 'te/genoie/ctranslate_clp', 'te/gen2oie_s1/clp', 'te/gen2oie_s1/translate_clp', 'te/gen2oie_s1/ctranslate_clp', 'te/gen2oie_s2/clp', 'te/gen2oie_s2/translate_clp', 'te/gen2oie_s2/ctranslate_clp', 'te/genoie/rerank', 'te/gen2oie_s1/ctranslate_clp_wopos', 'te/gen2oie_s1/translate_clp_wopos', 'te/gen2oie_s1/clp_wopos', 'te/gen2oie_s2/ctranslate_clp_wopos', 'te/gen2oie_s2/translate_clp_wopos', 'te/gen2oie_s2/clp_wopos']
# datasets += ['hi/gen2oie_s2/ctranslate_clp_wopos', 'hi/gen2oie_s2/translate_clp_wopos', 'hi/gen2oie_s2/clp_wopos', 'hi/gen2oie_s1/ctranslate_clp_wopos']
# datasets += ['es/gen2oie_s2/ctranslate_clp_wopos', 'es/gen2oie_s2/translate_clp_wopos', 'es/gen2oie_s2/clp_wopos', 'es/gen2oie_s1/ctranslate_clp_wopos']
# datasets += ['zh/gen2oie_s2/ctranslate_clp_wopos', 'zh/gen2oie_s2/translate_clp_wopos', 'zh/gen2oie_s2/clp_wopos', 'zh/gen2oie_s1/ctranslate_clp_wopos']
# datasets += ['en/gen2oie_s2/ctranslate_clp_wopos', 'en/gen2oie_s1/ctranslate_clp_wopos']
# datasets += ['pt/mt5/en', 'hi/mt5/en', 'zh/mt5/en', 'te/mt5/en', 'es/mt5/en']

datasets = ['hi_gen2oie_s1_aact_moie', 'hi_gen2oie_s2_aact_moie', 'te_gen2oie_s1_aact_moie', 'te_gen2oie_s2_aact_moie']
datasets += ['es_gen2oie_s1_aact_moie', 'es_gen2oie_s2_aact_moie', 'pt_gen2oie_s1_aact_moie', 'pt_gen2oie_s2_aact_moie']
datasets += ['zh_gen2oie_s1_aact_moie', 'zh_gen2oie_s2_aact_moie', 'en_gen2oie_s1', 'en_gen2oie_s2']

for dataset in datasets:
    t5.data.TaskRegistry.add(
        dataset,
        # Supply a function which returns a tf.data.Dataset.
        dataset_fn=functools.partial(dataset_fn, dataset=dataset),
        splits=["train", 'valid', 'test'],
        # Supply a function which preprocesses text from the tf.data.Dataset.
        text_preprocessor=preprocess,
        # Lowercase targets before computing metrics.
        postprocess_fn=postprocessor,
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[metrics.squad],
    )

t5.data.MixtureRegistry.add(f"Portuguese_openIE", [d.replace('/','_') for d in datasets], default_rate=1.0)
