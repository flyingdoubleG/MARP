import litellm
from pathlib import Path
from tqdm import tqdm

GENERATE_ESSAY_PROMPT_TEMPLATE = "Based on premise: \"{}\" generate story containing several scenes, use scene1:, scene2:, ... to represent."
RATE_ESSAY_PROMPT_TEMPLATE="Based on 1. Interesting. Interesting to the reader. 2. Coherent. Plot-coherent. 3. Relevant. Faithful to the initial premise. 4. Humanlike. Judged to be human-written.4 dimensions evaluate following 2 stories, the score is from 0 to 100, higher score means better.\nThe initial premise of story is \"{}\"\nStory 1: {}\n Story 2: {}."

QUARREL_PREMISE = "You will collaborate to create a story. The general setting: A Quarrel between two good friends about Iron Man."
IBRUSIA_PREMISE = "You will collaborate to create a story. The general setting: The state of Ibrusia is coming to a desperate and dangerous situation as the Hosso Union approaches its capital, Zaragoza."
ECONOMY_PREMISE = "You will collaborate to create a story. The general setting: The state of Gurata is coming to a huge economic recession. People are in panic and streets are in turmoil."


def get_response(model, message):
    response = litellm.completion(
        model=model,
        messages=[{"content": message, "role": "user"}],
    )
    return response.choices[0].message.content


def generate_baseline(baseline_model, premise, baseline_save_path):
    baseline = get_response(baseline_model, GENERATE_ESSAY_PROMPT_TEMPLATE.format(premise))
    with open(baseline_save_path, 'w+') as f:
        f.write(baseline)


def evaluate(essay_path, evaluator_model, premise, num_trials, baseline_path):
    with open(essay_path, 'r') as f:
        ours = f.read()
    with open(baseline_path, 'r') as f:
        baseline = f.read()
    evaluations = []
    for _ in tqdm(range(num_trials)):
        prompt = RATE_ESSAY_PROMPT_TEMPLATE.format(premise, ours, baseline)
        # if ('gpt-3.5' in evaluator_model or 'lama' in evaluator_model) and len(prompt) > 4096:
        #     print(f'truncating to 4096 tokens from {len(prompt)} tokens for {evaluator_model}')
        #     prompt = prompt[:4096]
        #     print(prompt)
        evaluation = get_response(evaluator_model, prompt)
        evaluations.append(evaluation)
    return evaluations


baseline_path = 'gemini-ibrusia.txt'
# generate_baseline('gemini-pro', IBRUSIA_PREMISE, baseline_path)
for evaluator_model in ['gpt-3.5-turbo-16k', 'gemini-pro', 'anyscale/mistralai/Mistral-7B-Instruct-v0.1']:
    print(f'Evaluating with {evaluator_model}...')
    evaluations = evaluate(
        essay_path='storys/mid_ibrusia.txt',
        evaluator_model=evaluator_model,
        premise=IBRUSIA_PREMISE,
        num_trials=5,
        baseline_path=baseline_path,
    )
    output_path = Path(f'evaluations/{evaluator_model}.txt')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w+') as f:
        f.write(f'********{evaluator_model}********\n')
        f.write('\n\n\n********************************\n\n\n'.join(evaluations))
