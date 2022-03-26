from transformers import T5Config, MT5ForConditionalGeneration, MT5Tokenizer
from transformers.models.t5.modeling_t5 import load_tf_weights_in_t5

if True:
    size = "base"
    model_name = f"google/mt5-{size}"
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration(T5Config.from_pretrained(model_name))

    dir_path = "/home/muqeeth101/moie_bucket/models/pt_gen2oie_s2_translate_clp"
    load_tf_weights_in_t5(model, None, dir_path)
    model.eval()

    model.save_pretrained(dir_path+'/hf_model')
    tokenizer.save_pretrained(dir_path+'/hf_model')
else:
    model_name = "persiannlp/mt5-small-parsinlu-sentiment-analysis"
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)

def run_model(context, query, **generator_args):
    input_ids = tokenizer.encode(context + "<sep>" + query, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    print(output)
    return output


