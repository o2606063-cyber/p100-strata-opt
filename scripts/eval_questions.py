#!/usr/bin/env python3
"""Run a fixed evaluation set against an OpenAI-compatible endpoint."""
import json
import sys
import time
import urllib.request

QUESTIONS = [
    {
        "id": "gsm8k_football",
        "q": "What is the total cost of purchasing equipment for all sixteen players on the football team, "
             "considering that each player requires a $25 jersey, a $15.20 pair of shorts, and a pair of socks "
             "priced at $6.80? Show your work briefly, then give the final number.",
        "answer": "752",
    },
    {
        "id": "logic_sisters",
        "q": "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have? "
             "Answer with a single number and one sentence of reasoning.",
        "answer": "2",
    },
    {
        "id": "code_even_odd",
        "q": "Write a Python function `even_odd_count(num)` that returns a tuple (even_count, odd_count) of the digits "
             "of the integer `num` (ignore the sign). Example: even_odd_count(-12) == (1, 1). Output only the function.",
        "answer": "CODE",
    },
    {
        "id": "strict_json",
        "q": "Output ONLY a JSON object, no prose, no code fences, with exactly these keys: "
             '"name" (string), "score" (integer), "tags" (array of 2 strings). Use name="demo", score=7, '
             'tags=["a","b"].',
        "answer": '{"name":"demo","score":7,"tags":["a","b"]}',
    },
    {
        "id": "chinese_sum",
        "q": "用中文回答：把下面这句话压缩成不超过 12 个字的结论，只输出结论本身。"
             "原句：由于内存带宽是这套双卡推理系统解码阶段的根本瓶颈，增加显卡数量并不能线性提升生成速度。",
        "answer": "语义判分",
    },
    {
        "id": "trap_strawberry",
        "q": "How many times does the letter 'r' appear in the word 'strawberry'? "
             "Answer with a single number and then the letters you counted.",
        "answer": "3",
    },
    {
        "id": "mmlu_style",
        "q": "Which of the following adaptive features would least likely be found in an animal living in a hot arid "
             "environment?\n(A) Long Loops of Henle to maximize water reabsorption\n"
             "(B) Storage of water in fatty tissues\n(C) Nocturnal behaviour to avoid daytime heat\n"
             "(D) Thick layer of subcutaneous blubber for insulation\nAnswer with the letter and one sentence why.",
        "answer": "D",
    },
]


def ask(base, model, question, max_tokens=3000):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    m = d["choices"][0]["message"]
    return {
        "content": m.get("content") or "",
        "reasoning": m.get("reasoning_content") or "",
        "prompt_tokens": d["usage"].get("prompt_tokens"),
        "completion_tokens": d["usage"].get("completion_tokens"),
        "finish": d["choices"][0].get("finish_reason"),
        "seconds": round(dt, 1),
    }


def main():
    base = sys.argv[1]
    model = sys.argv[2]
    out = sys.argv[3]
    results = []
    for item in QUESTIONS:
        print(f"  [{item['id']}] ...", flush=True)
        try:
            r = ask(base, model, item["q"])
        except Exception as e:
            r = {"error": str(e)}
        r["id"] = item["id"]
        r["expected"] = item["answer"]
        results.append(r)
        print(f"     {r.get('completion_tokens')} tok / {r.get('seconds')}s", flush=True)
    json.dump(results, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"  已保存 {out}")


main()
