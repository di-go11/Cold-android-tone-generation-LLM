import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "あなたは二十代の女性です。私とは友達関係です。"
text = "お腹すいたなーご飯でも食べに行こ"

model_path = "LLM/elyza/ELYZA-japanese-Llama-2-7b-fast-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

if torch.cuda.is_available():
    #model= model.half()
    # モデルを半精度に変換
    model = model.to("cuda")

prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
    bos_token=tokenizer.bos_token,
    b_inst=B_INST,
    system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
    prompt=text,
    e_inst=E_INST,
)


with torch.no_grad():
    inputs = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True  # attention maskを明示的に要求
    )

    output_ids = model.generate(
        input_ids=inputs["input_ids"].to(model.device),
        attention_mask=inputs["attention_mask"].to(model.device),
        max_new_tokens=256,  # 1024から256に減らす
        num_beams=1,        # ビームサーチを無効化
        do_sample=True,     # サンプリングを有効化
        temperature=0.7,    # 温度パラメータ
        top_k=40,          # top-k サンプリング
        top_p=0.9,         # top-p サンプリング
    )
output = tokenizer.decode(output_ids.tolist()[0][inputs["input_ids"].size(1) :], skip_special_tokens=True)
print(output)