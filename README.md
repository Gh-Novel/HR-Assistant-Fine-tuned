---
title: HR Assistant Fine-tuned
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.14.0"
python_version: "3.11"
app_file: app.py
pinned: false
license: llama3
short_description: QLoRA fine-tune of Llama 3.1 8B on TechMojo HR policies
---

# TechMojo HR Assistant — QLoRA Fine-tune of Llama 3.1 8B

[![HuggingFace Space](https://img.shields.io/badge/🤗%20Space-Live%20Demo-yellow)](https://huggingface.co/spaces/novelkathor/HR-Assistant-Fine-tuned)
[![GitHub](https://img.shields.io/badge/GitHub-Source-181717?logo=github)](https://github.com/Gh-Novel/HR-Assistant-Fine-tuned)
[![MLX](https://img.shields.io/badge/Apple_MLX-Framework-blue)](https://github.com/ml-explore/mlx)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)

A QLoRA fine-tune of **Llama 3.1 8B** on TechMojo's internal HR policies — an
**out-of-distribution** target chosen to demonstrate where fine-tuning genuinely
*creates capability* instead of restyling existing knowledge.

The base model has never seen TechMojo's specific procedures (Freshteams referral
portal, ADP for Form 16, 5-day carry-forward limit, 3-stakeholder unplanned-leave
escalation). We verified this with `data/techmojo/ood_check.py` before training:
the base model either hallucinates plausible-but-wrong details or admits ignorance.

🔗 **Try it live:** [HuggingFace Space — novelkathor/HR-Assistant-Fine-tuned](https://huggingface.co/spaces/novelkathor/HR-Assistant-Fine-tuned)
📦 **Source code:** [GitHub — Gh-Novel/HR-Assistant-Fine-tuned](https://github.com/Gh-Novel/HR-Assistant-Fine-tuned)

## 🖼️ Live Demo — Side by Side

The Gradio app runs both models on the same question. The fine-tuned panel sees the
TechMojo HR system prompt; the base panel sees a generic *"helpful assistant"* prompt
with **no mention of TechMojo** — that's the fair OOD comparison.

### Example 1 — Where to apply for leaves?

![Leave application question — base hallucinates a generic HR portal walkthrough; fine-tuned cites the actual TechMojo leave management URL](images/leaves-query.png)

The fine-tuned answer cites the exact TechMojo URL (`https://people.techmojo.com`)
and the policy line. The base model gives generic "1. HR website or portal /
2. HR office" guidance because it genuinely doesn't know the company.

### Example 2 — Reporting a technical issue

![Technical issue question — base lists generic troubleshooting steps; fine-tuned names the actual IT contacts and helpdesk email](images/technical-issue.png)

The fine-tuned answer names the actual TechMojo IT contacts
(*Mr. Naveen Peddi at 9966088494, Mr. Govardhan, it-helpdesk@techmojo.in*).
The base model walks through generic "Gather information / Check documentation /
Contact support" steps — useful generic advice, but not what an employee actually needs.

## 🎯 Key Results

<!-- BENCHMARK_START -->
## 📊 TechMojo HR Benchmark — Before vs After Fine-tuning

Held-out eval: TechMojo HR questions the base Llama 3.1 8B has never seen (the source dataset is small and not in pretraining corpora — verified via `data/techmojo/ood_check.py`).

| Metric | Base Llama 3.1 8B | Fine-tuned (QLoRA) | Δ |
|---|---|---|---|
| **Char-similarity to ground truth** | 0.034 | **0.166** | **+0.131** |
| **Keyword recall (TechMojo facts)** | 0.366 | **0.833** | **+0.468** |
| Examples | 18 | 18 | — |

**What this measures:** *Char-similarity* is `difflib.SequenceMatcher` between the model's response and the ground truth answer (0=no overlap, 1=identical text). *Keyword recall* is the fraction of TechMojo-specific facts (numbers, proper nouns, internal tool names like `Freshteams`/`ADP`) from the ground truth that appear in the model's response.

**Why both metrics:** char-similarity catches paraphrased correct answers; keyword recall catches whether the model knows the specific facts. Note that keyword recall can *favor verbose hallucination*: a rambling base-model answer mentions surface tokens like 'HR', 'manager', 'policies' by chance and scores high recall even when the factual content is wrong. The terse fine-tuned answer ('Yes, 1 week in advance') is correct but contains fewer total tokens to recall. Char-similarity is therefore the cleaner signal of factual correctness for this task.

### Side-by-side: 5 questions where fine-tuning helped most

**Q:** Who handles IT issues like system crashes or slow internet?

- **Ground truth (TechMojo):** If you're experiencing technical difficulties such as a system crash or slow internet, don't worry – we've got you covered. Our IT team is here to help resolve the issue quickly, and you can reach out to them directly for assistance. Whether it's a quick phone call to Mr. Naveen Peddi or Mr. Govardhan, or an email to our dedicated IT helpdesk, we're committed to getting you back up and running in no time.

**TechMojo policy:** Contact Mr. Naveen Peddi at 9966088494, Mr. Govardhan, or email it-helpdesk@techmojo.in for technical support.
- **Base Llama 3.1 8B:** Typically, IT issues such as system crashes or slow internet are handled by the IT department of an organization. This can include:

1. In-house IT teams: Many companies have their own IT departments that handle technical issues, provide support, and maintain the organization's technology infrastructure.
2. IT Help Desks: Some organizations have a dedicated help desk that employees can contact for assistance with IT-related issues.
3. IT Support Specialists: These are individuals who provide technical support and troubleshooting for IT-related issues.

In a home or personal setting, IT issues might be handled by:

1. The individual experiencing the issue themselves, by trying to troubleshoot and resolve the problem.
2. A family member or friend with technical expertise.
3. A third-party IT support service, such as a computer repair shop or a remote IT support provider.

It's worth noting that the specific process for handling IT issues can vary depending on the organization, company, or personal situation.
- **Fine-tuned:** If you're experiencing any IT issues, such as system crashes or slow internet, don't hesitate to reach out to our dedicated IT support team for assistance. They're here to help you get back up and running quickly, so you can focus on your work. You can contact them directly or email the IT helpdesk for prompt support.

**TechMojo policy:** Contact Mr. Naveen Peddi at 9966088494, Mr. Govardhan, or email it-helpdesk@techmojo.in for technical support.

  _char_similarity: 0.01 → 0.38 · keyword_recall: 0.50 → 1.00_

**Q:** Is there a deadline to apply for Flexi leave?

- **Ground truth (TechMojo):** To ensure that we can make the necessary arrangements for your time off, we need to have a heads up about when you're planning to take a flexi holiday. That's why we require at least 1 week's notice, so we can plan and schedule accordingly. This helps us to make the most of our team's time and ensures that everyone's needs are met.

**TechMojo policy:** Yes, you must apply for a flexi holiday at least 1 week in advance.
- **Base Llama 3.1 8B:** I don't have information about a specific company or organization called "Flexi leave." Could you please provide more context or details about what Flexi leave is or the company you are referring to? That way, I can try to provide a more accurate answer.
- **Fine-tuned:** To ensure that your team's plans and ours align, we require at least 1 week's notice for flexi leave. This allows us to make necessary arrangements and minimize disruptions. Please submit your flexi leave request to your supervisor at least 1 week in advance for approval.

**TechMojo policy:** Yes, you must apply for flexi leave at least 1 week in advance.

  _char_similarity: 0.04 → 0.29 · keyword_recall: 0.50 → 1.00_

**Q:** Can I take 3 Flexi Holidays in one month?

- **Ground truth (TechMojo):** To ensure that everyone gets a fair share of time off, we've set a limit on the number of Flexi Holidays that can be taken in a single month. This allows us to maintain a balanced workload and meet our business commitments. By capping the number of Flexi Holidays at 2 per month, we can ensure that our team's responsibilities are covered and that everyone can enjoy their time off with minimal disruption.

**TechMojo policy:** No, you are not eligible for more than 2 flexi holidays in a month.
- **Base Llama 3.1 8B:** I can provide general information about Flexi Holidays. However, I would like to clarify that I'm not aware of any specific company or organization that offers a program called "Flexi Holidays." 

If you're referring to a company that offers a flexible holiday policy, such as Virgin Holidays, I can try to provide general information about their policies. However, I would recommend checking directly with the company for the most up-to-date and accurate information.

That being said, if you're asking about a hypothetical or general scenario, I can try to provide a general answer. Typically, a flexible holiday policy allows employees to take a certain number of days off during the year, with some flexibility to take those days at different times.

However, taking 3 Flexi Holidays in one month might be an unusual scenario, and it's likely that there would be some restrictions or limitations on doing so. Without more information about the specific company or policy, it's difficult for me to provide a definitive answer.

If you
- **Fine-tuned:** To ensure fairness and transparency in our leave policy, we require that at least one business day separate each Flexi Holiday. This means that taking three Flexi Holidays in the same month would not be allowed, as it would not provide a fair distribution of time off throughout the month. This policy helps us maintain a balanced work schedule and ensures that our teams are always adequately staffed.

**TechMojo policy:** No, you cannot take 3 Flexi holidays in the same month.

  _char_similarity: 0.05 → 0.25 · keyword_recall: 0.50 → 1.00_

**Q:** Who approves Flexi Holidays?

- **Ground truth (TechMojo):** Flexi holidays are a great perk that allows you to take a day off when you need it most, but to ensure fairness and avoid last-minute rushes, we have a system in place where the Support Lead reviews and approves these requests on a case-by-case basis. This means that while you can't always count on getting a flexi holiday right away, the Support Lead will carefully consider each request to make sure everyone gets a fair turn. By having the Support Lead review these requests, we can balance everyone's needs and make sure this benefit is enjoyed by as many people as possible.

**TechMojo policy:** The Support Lead approves flexi holidays at their discretion. It's not always on a first-come, first-served basis to ensure fairness.
- **Base Llama 3.1 8B:** I don't have specific information about the approval process for Flexi Holidays. However, I can tell you that Flexi Holidays is a type of leave policy that allows employees to take a certain number of days off during the year, which can be used at any time. This type of policy is often offered by companies to provide employees with more flexibility and work-life balance.

If you're looking for information on who approves Flexi Holidays in a specific company, I would recommend checking the company's HR policies or speaking with a HR representative. They would be able to provide you with the most accurate and up-to-date information.
- **Fine-tuned:** To ensure that everyone has a fair chance to plan their vacations, the Support Lead has the final say in approving flexi holidays. This approach helps to balance everyone's needs and avoid last-minute changes. By making these decisions, the Support Lead aims to create a more predictable and manageable work environment for the team.

**TechMojo policy:** The Support Lead approves flexi holidays at their discretion.

  _char_similarity: 0.05 → 0.24 · keyword_recall: 0.50 → 1.00_

**Q:** Where do I declare taxes and view my payslips?

- **Ground truth (TechMojo):** As a TechMojo employee, it's essential to keep track of your income and taxes to ensure you're meeting your financial obligations. Once you've completed your first month with us, you'll need to use the ADP portal to declare your taxes and submit them for confirmation - this helps us ensure everything is accurate and up-to-date. This process is usually straightforward and can be completed through the ADP channel, which you can access through our internal systems.

**TechMojo policy:** You can view your payslips and declare taxes in the ADP channel. After your first month, you will need to declare taxes through the ADP portal and submit them for email confirmation.
- **Base Llama 3.1 8B:** The process of declaring taxes and viewing payslips can vary depending on your location and the specific tax authority in your country. Here are some general steps:

1. **Tax Authority Website**: In most countries, you can find the tax authority's website by searching online for "tax authority" or "tax office" along with your country's name. For example, in the United States, it's the IRS (Internal Revenue Service), while in the UK, it's HMRC (Her Majesty's Revenue and Customs).

2. **Payslip and Tax Declaration Portal**: Once you're on the tax authority's website, look for a portal or section that allows you to view your payslips and declare taxes. This might be a dedicated portal for employees or a general tax portal.

3. **Employee Portal**: If you're employed, you might have an employee portal provided by your employer. This portal usually allows you to view your payslips, tax deductions, and other employment-related information
- **Fine-tuned:** To declare your taxes and view your payslips, you'll need to log in to the ADP portal, which is our payroll system. This will allow you to review your payslips, declare your taxes, and even submit any necessary tax forms. By accessing the ADP portal, you'll be able to stay on top of your financial information and ensure everything is accurate.

**TechMojo policy:** You can view your payslips and declare taxes in the ADP portal.

  _char_similarity: 0.04 → 0.22 · keyword_recall: 0.00 → 0.67_


> Evaluated 18 held-out questions from `PranavTM/LeavePolicy` (TechMojo internal HR policies).  
> Base model: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`.  
> Fine-tuned with QLoRA (r=16, α=16) via Apple MLX on Mac Mini 24GB.  
> Eval timestamp: 2026-05-08

<!-- BENCHMARK_END -->

## 🏗 Project Structure

```
.
├── config.yaml                # All hyperparameters
├── requirements.txt
├── .env.example               # Copy to .env (HF_TOKEN, optional WANDB_API_KEY)
├── app.py                     # Gradio side-by-side demo (base vs fine-tuned)
├── Dockerfile                 # For HuggingFace Spaces
│
├── data/
│   ├── prepare_dataset.py     # Convert raw TechMojo Q&A → mlx-lm chat format
│   └── techmojo/
│       ├── raw.jsonl          # Source data (PranavTM/LeavePolicy)
│       └── ood_check.py       # Probe base model on specific questions before training
│
├── training/
│   ├── train.py               # QLoRA training (mlx_lm.lora wrapper)
│   └── callbacks.py           # W&B + local-file logging
│
├── evaluation/
│   ├── eval_qa.py             # Free-form QA eval (char-similarity + keyword recall)
│   └── compare.py             # Generate before/after report + update README
│
├── inference/
│   ├── chat.py                # Interactive CLI
│   └── api.py                 # FastAPI REST endpoint
│
└── export/
    └── push_to_hub.py         # Upload adapter weights to HuggingFace Hub (~40 MB)
```

## 🚀 Pipeline

```bash
# 1. Setup (one-time)
pip install -r requirements.txt
cp .env.example .env             # add HF_TOKEN if needed

# 2. Confirm the dataset is OOD for the base model (the whole reason this project works)
.venv/bin/python data/techmojo/ood_check.py

# 3. Prepare the training data
.venv/bin/python data/prepare_dataset.py

# 4. Score the base model BEFORE training (the "before" number)
.venv/bin/python evaluation/eval_qa.py --which base

# 5. Smoke test (50-iter, must see loss decrease)
.venv/bin/python training/train.py --test-run

# 6. Full training (~30-60 min on Mac Mini 24GB)
.venv/bin/python training/train.py

# 7. Score fine-tuned model
.venv/bin/python evaluation/eval_qa.py --which finetuned

# 8. Generate before/after report → updates this README
.venv/bin/python evaluation/compare.py

# 9. Demo
.venv/bin/python app.py            # Gradio side-by-side at localhost:7860
.venv/bin/python inference/chat.py # CLI

# 10. Publish
.venv/bin/python export/push_to_hub.py --dry-run    # preview model card
.venv/bin/python export/push_to_hub.py              # push adapters (~40 MB)
```

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 16 |
| LoRA Layers | 16 (top half of 32-layer Llama) |
| Trainable Params | ~0.8% of total |
| Epochs | 30 |
| Learning Rate | 1e-4 |
| Scheduler | Cosine + 30-step warmup |
| Batch Size | 1 (gradient_accumulation=8 → effective batch 8) |
| Hardware | Apple Silicon Mac Mini 24 GB |
| Framework | Apple MLX |

## 📚 Source Dataset

| Dataset | Size | Source |
|---|---|---|
| `PranavTM/LeavePolicy` | 117 examples | HuggingFace |

The dataset covers TechMojo's leave, sabbatical, referral, payroll, and manager
escalation policies — internal procedures the base Llama model has no exposure
to in pretraining. This is the project's central design choice: a fine-tune
target that's *actually out of distribution*.

## 💬 Inference

```bash
# Interactive CLI
.venv/bin/python inference/chat.py

# Compare base vs fine-tuned for the same question
.venv/bin/python inference/chat.py --compare

# REST API
.venv/bin/uvicorn inference.api:app --reload --port 8000

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many leave days do I get per year at TechMojo?"}'
```

## ⚠️ Disclaimer

This is a research / portfolio project. The fine-tuning data is the public
`PranavTM/LeavePolicy` dataset on HuggingFace, which describes a specific
fictional/illustrative company. This is **not** real HR advice for any actual
employer.

## 📄 License

- Code: MIT
- Model weights: [Meta Llama 3 Community License](https://llama.meta.com/llama3/license/)
- Dataset: per [`PranavTM/LeavePolicy`](https://huggingface.co/datasets/PranavTM/LeavePolicy) terms
