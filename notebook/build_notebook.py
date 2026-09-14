"""Generates notebook/finetune.ipynb. Run after editing: python notebook/build_notebook.py"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

CELLS = [
    ("md", """# Fine-tune GLM-5.3 privately, on a full node, inside an enclave

This notebook is running inside a [Tinfoil Container](https://docs.tinfoil.sh/containers/overview): a hardware-attested enclave holding all eight GPUs of the machine. Three things are true here that are not true on an ordinary GPU box:

- **The base model is verified.** GLM-5.3 (744B parameters, 256 routed experts per layer) is mounted read-only from a model pack whose root hash is part of the enclave measurement. It is the exact checkpoint Tinfoil serves.
- **Your data and your adapter stay private.** `/workspace` is a 16 TB encrypted, integrity-protected disk. It was unlocked at boot with a key released only to this measured enclave, and it survives restarts and updates.
- **Nothing leaves.** The enclave has no network egress. The only way in or out is this notebook, over the attested TLS connection you are using right now.

Run the cells top to bottom (**Run ▸ Run All Cells**). Loading the 465 GB checkpoint takes several minutes; training on the sample data takes a few more."""),

    ("code", '''import json, math, os, random, shutil, time
from pathlib import Path

import torch, yaml

MODEL_DIR = os.environ["MODEL_DIR"]
WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))

config = yaml.safe_load(Path("/tinfoil/config.yml").read_text())
volume = config["volumes"][0]
mount = next(line.split() for line in Path("/proc/self/mountinfo").read_text().splitlines() if line.split()[4] == str(WORKSPACE))
usage = shutil.disk_usage(WORKSPACE)

print("cvm-version :", config["cvm-version"])
print("base model  :", config["models"][0]["repo"])
print("workspace   :", f"volume '{volume['name']}' unlocked at boot with secret {volume['key-secret']}")
print("mounted from:", mount[-2], f"({mount[-3]}), {usage.total / 1e12:.1f} TB, {usage.free / 1e12:.1f} TB free")
print("egress      :", "none (no `networks:` in the measured config)" if not config.get("networks") else config["networks"])
print("GPUs        :", torch.cuda.device_count(), "x", torch.cuda.get_device_name(0), f"({torch.cuda.mem_get_info(0)[1] / 2**30:.0f} GiB each)")'''),

    ("md", """## 1. Load the verified base model across the eight GPUs

The weights come from `/tinfoil/mpk/...`, a read-only mount of the model pack pinned in `tinfoil-config.yml`. Nothing is downloaded: the enclave could not reach Hugging Face even if it wanted to.

The checkpoint keeps the 256 routed experts of every MoE layer in NVFP4 (4-bit weights with per-block scales), which is how the model is served. `glm_nvfp4.py`, shipped in this image, streams it into transformers' `GlmMoeDsaForCausalLM` with the decoder layers split across the GPUs and the experts left packed; they are dequantized on the fly inside each forward pass (a small `torch.compile`d kernel), so gradients flow through them and the whole model fits in about 55 GB per GPU. Loading takes about five minutes the first time and about one minute once the pack is in the page cache."""),

    ("code", '''from transformers import AutoTokenizer
from glm_nvfp4 import load_model, memory_report

started = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = load_model(MODEL_DIR)
print(f"\\nloaded in {time.time() - started:.0f}s; {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B bf16 parameters "
      f"+ {sum(b.numel() for n, b in model.named_buffers() if n.endswith('_w')) * 2 / 1e9:.0f}B packed FP4 expert weights")
print(memory_report())

THINK_OPEN, THINK_CLOSE, TURN_END = "<think>", "</think>", "<|user|>"


def ask(question, max_new_tokens=64):
    """Greedy answer to a single user turn, with the thinking block left empty so the model answers directly."""
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
    assert prompt.endswith(THINK_OPEN)
    inputs = tokenizer(prompt + THINK_CLOSE, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()'''),

    ("md", """## 2. Your private data

Training data is a JSONL file of chat conversations, one per line: `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`. Drop your own file into `data/` using the file browser on the left; it lands on the encrypted volume and never touches the host. With 16 TB there is room for real datasets and for checkpoints.

The sample, `data/train.jsonl`, is an internal helpdesk for a fictional coffee company. Every fact in it is invented, so the base model cannot know any of them. That makes the before/after obvious."""),

    ("code", '''def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

train_rows = read_jsonl(WORKSPACE / "data" / "train.jsonl")
eval_rows = read_jsonl(WORKSPACE / "data" / "eval.jsonl")
print(f"{len(train_rows)} training conversations, {len(eval_rows)} held-out questions\\n")
for row in train_rows[:2]:
    print("user     :", row["messages"][0]["content"])
    print("assistant:", row["messages"][1]["content"].replace("\\n", " "), "\\n")'''),

    ("md", """### Before training

Ask the base model three of the held-out questions. It has never seen this company, so it guesses. (Each answer is a full pass through 744B parameters on eight GPUs; expect about a minute per question.)"""),

    ("code", '''EVAL_QUESTIONS = [row["messages"][0]["content"] for row in eval_rows[:3]]
for question in EVAL_QUESTIONS:
    started = time.time()
    print(f"Q: {question}\\nA: {ask(question)}   [{time.time() - started:.0f}s]\\n")'''),

    ("md", """## 3. Tokenize

Each conversation is rendered with the model's chat template. GLM-5.3 is a reasoning model: the template opens a `<think>` block before every answer, and the training examples close it immediately so the tuned model answers directly. Loss is computed on the assistant's turn only: the prompt tokens get the label `-100`, which PyTorch ignores. The `<|user|>` token that ends an assistant turn is appended so the model learns where to stop."""),

    ("code", '''TURN_END_ID = tokenizer.convert_tokens_to_ids(TURN_END)


def encode(row):
    messages = row["messages"]
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    assert full.startswith(prompt) and THINK_OPEN + THINK_CLOSE in full[len(prompt) - len(THINK_OPEN):], "unexpected chat template"
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False)["input_ids"] + [TURN_END_ID]
    labels = [-100] * len(prompt_ids) + ids[len(prompt_ids):]
    return {"input_ids": ids, "labels": labels}

encoded = [encode(row) for row in train_rows]
lengths = [len(e["input_ids"]) for e in encoded]
print(f"{len(encoded)} examples, {min(lengths)}-{max(lengths)} tokens each")
print(repr(tokenizer.decode(encoded[0]["input_ids"])))


def collate(batch):
    width = max(len(e["input_ids"]) for e in batch)
    pad = tokenizer.pad_token_id
    input_ids = torch.tensor([e["input_ids"] + [pad] * (width - len(e["input_ids"])) for e in batch])
    labels = torch.tensor([e["labels"] + [-100] * (width - len(e["labels"])) for e in batch])
    attention_mask = torch.tensor([[1] * len(e["input_ids"]) + [0] * (width - len(e["input_ids"])) for e in batch])
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}'''),

    ("md", """## 4. Attach LoRA adapters

LoRA trains a small number of extra parameters on top of the frozen base model. Here the adapters attach to the attention projections (GLM-5.3 uses multi-head latent attention, hence the `q_a`/`q_b`/`kv_a`/`kv_b` names), to the dense MLPs of the first three layers and to the shared expert of every MoE layer. The 256 routed experts per layer stay packed in FP4 and untouched; gradients flow through them to the adapters below."""),

    ("code", '''from peft import LoraConfig, get_peft_model

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=r"model\\.layers\\.\\d+\\.(self_attn\\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)|mlp\\.(gate_proj|up_proj|down_proj)|mlp\\.shared_experts\\.(gate_proj|up_proj|down_proj))",
)
model = get_peft_model(model, lora)
for parameter in model.parameters():
    if parameter.requires_grad:
        parameter.data = parameter.data.float()  # keep the trainable weights in fp32; the base stays bf16 / FP4
model.print_trainable_parameters()'''),

    ("md", """## 5. Train

A plain PyTorch loop, so there is nothing hidden: forward, backward, clip, step. Each step runs the batch through the eight GPUs in turn and re-dequantizes the experts it touches, so expect a few seconds per step. The loss curve updates live."""),

    ("code", '''import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from transformers import get_linear_schedule_with_warmup

EPOCHS, BATCH_SIZE, LEARNING_RATE = 3, 4, 2e-4

trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.0)
steps_per_epoch = math.ceil(len(encoded) / BATCH_SIZE)
total_steps = EPOCHS * steps_per_epoch
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)

random.seed(0)
model.train()
losses, step, started = [], 0, time.time()
figure, axis = plt.subplots(figsize=(7, 3))
for epoch in range(EPOCHS):
    order = random.sample(range(len(encoded)), len(encoded))
    for start in range(0, len(order), BATCH_SIZE):
        batch = collate([encoded[i] for i in order[start:start + BATCH_SIZE]])
        batch = {k: v.to(model.device) for k, v in batch.items()}
        loss = model(**batch, use_cache=False).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
        step += 1
        axis.clear()
        axis.plot(losses)
        axis.set(xlabel="step", ylabel="loss", title=f"epoch {epoch + 1}/{EPOCHS}   step {step}/{total_steps}   loss {losses[-1]:.3f}   {(time.time() - started) / step:.1f}s/step")
        clear_output(wait=True)
        display(figure)
plt.close(figure)
model.eval()
print(f"{step} steps in {time.time() - started:.0f}s, loss {losses[0]:.2f} -> {losses[-1]:.3f}")'''),

    ("md", """## 6. Save the adapter to the encrypted workspace

`save_pretrained` writes the LoRA weights into `adapters/<run>/` on the encrypted volume. They stay there across container restarts and updates, and the host never sees them in the clear."""),

    ("code", '''run_dir = WORKSPACE / "adapters" / time.strftime("%Y%m%d-%H%M%S")
model.save_pretrained(run_dir)
(run_dir / "training.json").write_text(json.dumps({
    "base_model": config["models"][0]["repo"],
    "examples": len(encoded), "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
    "final_loss": losses[-1], "losses": losses,
}, indent=2))
for path in sorted(run_dir.iterdir()):
    print(f"{path.stat().st_size / 1e6:8.2f} MB  {path.relative_to(WORKSPACE)}")'''),

    ("md", """## 7. Before and after

Same questions, same weights, adapter off then on. Everything the tuned model knows about the company came from `data/train.jsonl`."""),

    ("code", '''for question in EVAL_QUESTIONS:
    with model.disable_adapter():
        before = ask(question)
    after = ask(question)
    print(f"Q: {question}\\n   base : {before}\\n   tuned: {after}\\n")'''),

    ("md", """## 8. It persists

Stop the container from your laptop, wait until it reports `stopped` (a 512 GB enclave takes about two minutes to shut down; a start issued while it is still stopping is refused), start it again, reconnect, and run the next cell in a fresh kernel. The adapter is still on the encrypted volume, and it loads onto the freshly verified base model.

```bash
tinfoil container stop finetune-glm
tinfoil container get finetune-glm      # repeat until STATUS is stopped
tinfoil container start finetune-glm
```"""),

    ("code", '''import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer
from glm_nvfp4 import load_model

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
runs = sorted(p for p in (WORKSPACE / "adapters").iterdir() if (p / "adapter_config.json").exists())
print("adapters on the encrypted volume:", *[run.name for run in runs], sep="\\n  ")

tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_DIR"])
tuned = PeftModel.from_pretrained(load_model(os.environ["MODEL_DIR"]), runs[-1]).eval()

prompt = tokenizer.apply_chat_template([{"role": "user", "content": "When does the Huila Reserve ship?"}], tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt + "</think>", add_special_tokens=False, return_tensors="pt").to(tuned.device)
with torch.no_grad():
    out = tuned.generate(**inputs, max_new_tokens=64, do_sample=False)
print("\\n" + tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())'''),

    ("md", """## Next steps

- **Bring real data.** The volume is 16 TB: upload datasets through the file browser, keep checkpoints, and tune the routed experts too with PEFT's `target_parameters` if you have the tokens to justify it.
- **Keep the key away from Tinfoil.** Set `keyserver-url` in `tinfoil-config.yml` and release `WORKSPACE_KEY` and `JUPYTER_TOKEN` from your own [keyserver](https://docs.tinfoil.sh/containers/private-secrets); the volume is then unreadable to the operator as well.
- **Start smaller.** The single-GPU [finetuning-example](https://github.com/tinfoilsh/finetuning-example) runs the same notebook against Gemma 4 E2B in about three minutes.
- **Verify from outside.** `tinfoil attestation verify -e <your-domain> -r <owner>/<repo>` checks that the enclave you are talking to runs exactly the measured release."""),
]


def build():
    notebook = nbf.v4.new_notebook()
    notebook.cells = [nbf.v4.new_markdown_cell(body) if kind == "md" else nbf.v4.new_code_cell(body) for kind, body in CELLS]
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    notebook.metadata["language_info"] = {"name": "python"}
    nbf.validate(notebook)
    nbf.write(notebook, HERE / "finetune.ipynb")
    print(f"wrote {HERE / 'finetune.ipynb'} ({len(CELLS)} cells)")


if __name__ == "__main__":
    build()
