"""Evidence-constrained Hindi generation with citations selected from retrieved metadata."""
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import GENERATION_MODEL, MAX_CONTEXT_CHARS, TORCH_NUM_THREADS, MAX_NEW_TOKENS, RAG_DEVICE

class HindiGroundedGenerator:
    def __init__(self):
        torch.set_num_threads(TORCH_NUM_THREADS)
        device = RAG_DEVICE if RAG_DEVICE.startswith('cuda') and torch.cuda.is_available() else 'cpu'
        dtype = torch.float16 if device != 'cpu' and torch.cuda.is_available() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            GENERATION_MODEL, dtype=dtype, low_cpu_mem_usage=True
        ).to(device)
        self.model.eval()
        self.device = device

    def answer(self, question, contexts):
        if not contexts:
            return 'दिए गए दस्तावेज़ में इस प्रश्न का समर्थन करने वाला साक्ष्य नहीं मिला।'
        evidence, used = [], 0
        usable = [c for c in contexts if c.get('source_type') != 'practice_question']
        for c in usable:
            part = (f"[पृष्ठ {c['page']} | अनुभाग {c.get('section','दस्तावेज़')} | chunk {c['chunk_id']}]\n{c['text']}")
            if used + len(part) > MAX_CONTEXT_CHARS:
                break
            evidence.append(part); used += len(part)
        if not evidence:
            return 'दिए गए दस्तावेज़ में इस प्रश्न का समर्थन करने वाला साक्ष्य नहीं मिला।'
        system = (
            'आप हिंदी दस्तावेज़-आधारित प्रश्नोत्तर सहायक हैं। केवल नीचे दिए गए साक्ष्य का उपयोग करें। '
            'उत्तर देवनागरी हिंदी में संक्षिप्त और सीधे दें, भले ही प्रश्न अंग्रेज़ी में हो। '
            'सामान्य ज्ञान न जोड़ें। प्रश्न-सूची या अभ्यास-प्रश्न को तथ्यात्मक प्रमाण न मानें। '
            'यदि साक्ष्य पर्याप्त न हो तो स्पष्ट कहें। पृष्ठ संख्या या chunk ID स्वयं न लिखें; citations सिस्टम जोड़ेगा।'
        )
        evidence_text = '\n\n'.join(evidence)
        user = f"प्रश्न: {question}\n\nदस्तावेज़ से प्राप्त साक्ष्य:\n{evidence_text}\n\nउत्तर (हिंदी में):"
        prompt = self.tokenizer.apply_chat_template(
            [{'role':'system','content':system},{'role':'user','content':user}],
            tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                repetition_penalty=1.05, pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id)
        generated = output[0][inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def select_citations(question: str, answer: str, contexts: list[dict], limit: int = 3) -> list[dict]:
    """Return actual retrieved source metadata; never accept citations authored by the LLM."""
    candidates = [c for c in contexts if c.get('source_type') != 'practice_question']
    if not candidates:
        return []
    # Keep high-ranked evidence; prefer passages that overlap with key answer tokens.
    terms = set(re.findall(r'[\u0900-\u097F]{2,}|[A-Za-z0-9]{3,}', answer.casefold()))
    def rank(c):
        text_terms = set(re.findall(r'[\u0900-\u097F]{2,}|[A-Za-z0-9]{3,}', c['text'].casefold()))
        overlap = len(terms & text_terms) / max(1, len(terms))
        return (0.65 * float(c.get('score', 0)) + 0.35 * overlap)
    ranked = sorted(candidates, key=rank, reverse=True)
    selected = []
    for c in ranked:
        if c['chunk_id'] not in {x['chunk_id'] for x in selected}:
            selected.append(c)
        if len(selected) >= limit:
            break
    return [{'page': int(c['page']), 'section': c.get('section','दस्तावेज़'),
             'chunk_id': c['chunk_id'], 'score': round(float(c.get('score',0)),5),
             'source_type': c.get('source_type','paragraph'), 'text': c['text']} for c in selected]
