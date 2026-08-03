"""Student side: QLoRA SFT on accepted teacher traces, generation and scoring.

Heavy imports (torch, transformers, peft, trl, datasets) happen inside the
functions so the package imports cleanly before `pip install` has run.
"""
from .synth import COND_BOUNDS, FAULTS
from .verify import check, parse_answer

_C0, _C1, _C2 = COND_BOUNDS

# The student prompt must carry the answer vocabulary — otherwise the base model
# answers in free text ("inner race bearing fault") and exact-match scoring
# reads as ~0 even when the diagnosis is right.
STUDENT_SYSTEM = (
    "You are a vibration analysis and predictive-maintenance expert. "
    "Analyse the measurement report step by step under a heading 'Analysis:': "
    "infer the true running speed from the spectrum, identify the fault "
    f"(exactly one of: {', '.join(FAULTS)}), the bearing designation if unknown, the "
    "ISO 20816-3 zone (A/B/C/D) from the boundaries stated in the report, and "
    "the bearing condition from the largest envelope-spectrum peak "
    f"(none < {_C0} g <= early < {_C1} g <= moderate < {_C2} g <= severe). "
    "Then output one fenced json block with keys: inferred_speed_rpm, fault_type, "
    "iso_zone, bearing_condition, bearing_designation, key_evidence, "
    "recommended_actions."
)


def pick_student():
    """Qwen3-8B QLoRA fits comfortably in 40 GB (A100); Qwen3-4B in 24 GB (L4)."""
    import torch
    assert torch.cuda.is_available(), "No GPU — Runtime > Change runtime type > A100 (or L4)"
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30
    name = "Qwen/Qwen3-8B" if vram >= 36 else "Qwen/Qwen3-4B"
    print(f"{torch.cuda.get_device_name(0)} — {vram:.0f} GB VRAM -> student {name}")
    return name, vram


def load_student(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        name, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.config.use_cache = True
    print(f"{name} loaded — {sum(p.numel() for p in model.parameters()) / 1e9:.1f} B params (4-bit)")
    return model, tok


def build_sft_dataset(accepted, seed):
    from datasets import Dataset

    records = [{"messages": [
        {"role": "system", "content": STUDENT_SYSTEM},
        {"role": "user", "content": a["sample"]["report"]},
        {"role": "assistant", "content": a["text"]},
    ]} for a in accepted]
    return Dataset.from_list(records).shuffle(seed=seed).train_test_split(test_size=0.03, seed=seed)


def train_student(model, tok, ds, cfg, vram_gb):
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    peft_cfg = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    bs = 4 if vram_gb >= 36 else 2          # effective batch stays 16 via accumulation
    args = SFTConfig(
        output_dir=f"{cfg.out_dir}/student",
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=16 // bs,
        learning_rate=1e-4, lr_scheduler_type="cosine", warmup_steps=20,
        logging_steps=10, eval_strategy="steps", eval_steps=50, save_strategy="epoch",
        bf16=True, gradient_checkpointing=True,
        max_length=3072,          # older TRL (<0.13): rename to max_seq_length
        packing=False, report_to="none",
    )
    trainer = SFTTrainer(model=model, args=args, peft_config=peft_cfg,
                         train_dataset=ds["train"], eval_dataset=ds["test"],
                         processing_class=tok)
    trainer.train()
    return trainer


def generate_batch(mdl, tok, prompts, bs=8, max_new_tokens=1200):
    import torch
    from tqdm.auto import tqdm

    outs = []
    mdl.eval()
    for i in tqdm(range(0, len(prompts), bs), desc="generating"):
        chats = [tok.apply_chat_template(
            [{"role": "system", "content": STUDENT_SYSTEM}, {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ) for p in prompts[i:i + bs]]
        enc = tok(chats, return_tensors="pt", padding=True).to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id, use_cache=True)
        outs += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


def score(samples, outputs):
    """Accuracy over ALL samples — an unparseable answer counts as wrong."""
    import pandas as pd

    res = [check(s, parse_answer(o)) for s, o in zip(samples, outputs)]
    df = pd.DataFrame(res)
    return {"parse": df["parsed"].mean(), "fault": df["fault"].mean(), "zone": df["zone"].mean(),
            "cond": df["cond"].mean(), "speed<=4%": df["speed"].mean()}


def save_adapter(trainer, tok, cfg, student_name):
    final_dir = f"{cfg.out_dir}/vib-expert-{student_name.split('/')[-1]}-lora"
    trainer.model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)
    print("LoRA adapter saved to", final_dir)
    return final_dir
