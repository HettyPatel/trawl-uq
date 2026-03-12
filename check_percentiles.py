import pickle, numpy as np
d = pickle.load(open('results/sae_uncertainty_mmlu14k/sae_uncertainty_Llama-3.1-8B.pkl','rb'))
ents = [q['entropy'] for q in d['per_question_data']]
for p in [10, 20, 25, 33]:
    lo = np.percentile(ents, p)
    conf = [q for q in d['per_question_data'] if q['entropy'] <= lo]
    B = [q for q in conf if not q['correct']]
    print(f'p={p}%: conf={len(conf)}, B={len(B)}, B/conf={100*len(B)/len(conf):.1f}%')
