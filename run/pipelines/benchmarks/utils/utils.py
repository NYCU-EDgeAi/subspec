import hashlib
import jieba
import re
import string
import torch

from collections import Counter
from fuzzywuzzy import fuzz
from rouge import Rouge
from typing import List, Tuple, Dict, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r'Paragraph (\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def retrieval_zh_score(prediction, ground_truth, **kwargs):
    pattern = r'段落(\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def code_sim_score(prediction, ground_truth, **kwargs):
    all_lines = prediction.lstrip('\n').split('\n')
    prediction = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            prediction = line
            break
    return (fuzz.ratio(prediction, ground_truth) / 100)

def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        score = (1.0 / len(em_match_list))
    else:
        score = 0.0
    return score
    
def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]

def rouge_zh_score(prediction, ground_truth, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False))) 
    score = rouge_score(prediction, ground_truth)
    return score

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)


# for Longgenbench
class LLMJudge:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: Optional[str] = None,
        device_map: Optional[str] = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
        max_new_tokens: int = 2,
    ):
        if torch_dtype is None:
            torch_dtype = (torch.bfloat16
                           if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
                           else torch.float16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if device is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype
            ).to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype,
                device_map=device_map
            )
        self.model.eval()

        self.max_new_tokens = max_new_tokens
        self._cache: Dict[str, bool] = {}

    def judge(self, context: str, event_desc: str) -> bool:
        key = hashlib.md5((context + "||" + event_desc).encode("utf-8")).hexdigest()
        if key in self._cache:
            return self._cache[key]

        # build chat prompt
        msgs = _make_prompt(context, event_desc)
        prompt = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        enc = self.tokenizer(prompt, return_tensors="pt")
        first_param = next(self.model.parameters())
        enc = {k: v.to(first_param.device) for k, v in enc.items()}

        with torch.no_grad():
            out_ids = self.model.generate(
                **enc,
                do_sample=False, temperature=0.0,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        input_len = enc["input_ids"].shape[1]
        gen_ids = out_ids[0, input_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        ans = _parse_yes_no(text)
        self._cache[key] = ans
        return ans

    def judge_batch(self, pairs: List[Tuple[str, str]]) -> List[bool]:
        return [self.judge(c, e) for (c, e) in pairs]

def _make_prompt(context: str, event_desc: str) -> str:
    messages = []
    messages += [
        {"role": "user", "content":
        "Context: The district's new residential area ... medical clinic.\n"
        "Instruction: Does this context include a medical clinic? Please answer with 'yes' or 'no' only."},
        {"role": "assistant", "content": "yes"},
        {"role": "user", "content":
        "Context: The menu for today's lunch ... mashed potatoes ...\n"
        "Instruction: Does this context include mashed potatoes? Please answer with 'yes' or 'no' only."},
        {"role": "assistant", "content": "yes"},
        {"role": "user", "content":
        "Context: On April 15th ... planted 20 trees ...\n"
        "Instruction: Does this context include long-distance running? Please answer with 'yes' or 'no' only."},
        {"role": "assistant", "content": "no"},
    ]

    messages.append({
        "role": "user",
        "content": (
            f" \n ### Refer to the examples above for how to answer. \nContext: {context}"
            f"\n\n### Instruction: Does this context include {event_desc}? "
            f"Please answer with 'yes' or 'no' only."
        )
    })

    return messages

def _parse_yes_no(text: str) -> bool:
    m = re.compile(r"^\s*(yes|no)\b", re.I).search(text.strip())
    if m:
        return m.group(1).lower() == "yes"
    t = text.lower()
    if ("yes" in t) ^ ("no" in t):
        return "yes" in t
    return t.startswith("yes")

def _as_phrase_list(v) -> List[str]:
    """Normalize any value into List[str]."""
    if v is None:
        return []
    if isinstance(v, str):
        v = v.strip()
        return [v] if v else []
    if isinstance(v, (list, tuple, set)):
        out = []
        for x in v:
            xs = str(x).strip()
            if xs:
                out.append(xs)
        return out
    s = str(v).strip()
    return [s] if s else []

def _split_sections_by_marker(text: str) -> List[str]:
    """ Split the text by #*# """
    parts = re.split(r"\s*#\*#\s*", text)
    parts = [p.strip() for p in parts if p and p.strip()]
    if parts and parts[0].lstrip('* ').lower().startswith('started'):
        parts = parts[1:]
    return parts

def _norm_checks_dict(d: Any) -> Dict[int, List[str]]:
    out = {}
    for k, v in d.items():
        ki = int(str(k))
        phrases = _as_phrase_list(v)
        if phrases:
            out[ki] = phrases
    return out

def _collect_pairs(sections: List[str], item: Dict[int, List[str]], clamp: int) -> List[Tuple[str, str, int]]:
    pairs = []
    for idx, phrases in item.items():
        if clamp and idx > clamp: continue
        ctx = sections[idx-1] if 1 <= idx <= len(sections) else ""
        for ph in (phrases if isinstance(phrases, list) else [phrases]):
            pairs.append((ctx, ph, idx))
    return pairs

def _llm_hits_from_pairs(pairs: List[Tuple[str, str, int]], N: int, judge) -> Tuple[int, int, int, int]:
    ctx_ev = [(c, e) for (c, e, _) in pairs]
    ys = judge.judge_batch(ctx_ev)
    by_index: Dict[int, bool] = {}
    by_N_index: Dict[int, bool] = {}

    for (c, e, idx), y in zip(pairs, ys):
        if idx<=N:
            by_N_index[idx] = by_N_index.get(idx, False) or bool(y)
        by_index[idx] = by_index.get(idx, False) or bool(y)

    M_hits = sum(1 for v in by_index.values() if v)
    M_denom = len(by_index)
    N_hits = sum(1 for v in by_N_index.values() if v)
    N_denom = len(by_N_index)

    return M_hits, M_denom, N_hits, N_denom

def score_longgenbench_single(output_str: str, meta: dict) -> dict:
    # split section 
    sections = _split_sections_by_marker(output_str)
    N = int(meta.get("number", 0)) if str(meta.get("number", "")).isdigit() else 0
    M = min(len(sections), N) if N > 0 else len(sections)

    si = _norm_checks_dict(meta.get("checks_once", {}))
    r_items = _norm_checks_dict(meta.get("checks_range", {}))
    p_items = _norm_checks_dict(meta.get("checks_periodic", {}))

    # scoring via LLM as a Judge
    llm_judge = LLMJudge()  
    clampN = N if N>0 else len(sections)

    si1_h, si1_d, si2_h, si2_d = _llm_hits_from_pairs(
        _collect_pairs(sections, si, M), clampN, llm_judge
    )
    r1_h, r1_d, r2_h, r2_d = _llm_hits_from_pairs(
        _collect_pairs(sections, r_items, M), clampN, llm_judge
    )
    p1_h, p1_d, p2_h, p2_d = _llm_hits_from_pairs(
        _collect_pairs(sections, p_items, M), clampN, llm_judge
    )
    
    # calculate STIC1 score
    stic1 = ( (si1_h + r1_h + p1_h) / (si1_d + r1_d + p1_d) ) if (si1_d + r1_d + p1_d)>0 else 0.0
    stic1_once = (si1_h / si1_d) if si1_d>0 else 0.0
    stic1_range = (r1_h / r1_d) if r1_d>0 else 0.0
    stic1_periodic = (p1_h / p1_d) if p1_d>0 else 0.0

    # calculate STIC2 score (clamp to N)
    stic2 = ( (si2_h + r2_h + p2_h) / (si2_d + r2_d + p2_d) ) if (si2_d + r2_d + p2_d)>0 else 0.0
    stic2_once = (si2_h / si2_d) if si2_d>0 else 0.0
    stic2_range = (r2_h / r2_d) if r2_d>0 else 0.0
    stic2_periodic = (p2_h / p2_d) if p2_d>0 else 0.0

    # calculate cr and wavg
    cr = (M / N) if N > 0 else (1.0 if len(sections) > 0 else 0.0)
    wavg = cr * stic2
    return {
        "M": int(M), "N": int(N), "cr": float(cr),
        "stic1_once": float(stic1_once), "stic1_range": float(stic1_range), "stic1_periodic": float(stic1_periodic),
        "stic1_overall": float(stic1),
        "stic2_once": float(stic2_once), "stic2_range": float(stic2_range), "stic2_periodic": float(stic2_periodic),
        "stic2_overall": float(stic2), "wavg": float(wavg),
        "denoms": {"stic1": {"once": int(si1_d), "range": int(r1_d), "periodic": int(p1_d), "total": int(si1_d+r1_d+p1_d)},
                   "stic2": {"once": int(si2_d), "range": int(r2_d), "periodic": int(p2_d), "total": int(si2_d+r2_d+p2_d)}}
    }