import litellm
from pathlib import Path
from tqdm import tqdm
import numpy as np
import copy
import csv
import re

GENERATE_ESSAY_PROMPT_TEMPLATE = "Based on premise: \"{}\" generate story containing several scenes, use scene1:, scene2:, ... to represent."

RATE_ESSAY_PROMPT_TEMPLATE="Based on 1. Interesting. Interesting to the reader. 2. Coherent. Plot-coherent. 3. Relevant. Faithful to the initial premise. 4. Humanlike. Judged to be human-written.4 dimensions evaluate following 2 stories, the score is from 0 to 100, higher score means better.\nThe initial premise of story is \"{}\"\nStory 1: {}\n Story 2: {}."

HANNA_RATE_ESSAY_PROMPT_TEMPLATE="Based on the following six categories: 1. Relevance. 2. Coherence. 3. Empathy. 4. Surprise. 5. Engagement. 6. Complexity, evaluate the following two stories by assigning an integer score (from 1 to 5) to each category. Higher score means better.\nThe initial premise of story is: {}\nStory1: {}\n Story2: {}.\n\nIn your response, please use the following example format strictly and no need for any extra explanations: \nStory1\nRelevance: \nCoherence: \nEmpathy: \nSurprise: \nEngagement: \nComplexity: \nStory2\nRelevance: \nCoherence: \nEmpathy: \nSurprise: \nEngagement: \nComplexity: \n"

# HANNA_RATE_ESSAY_PROMPT_TEMPLATE="Based on the following four categories: 1. Interesting, 2. Coherent, 3. Relevant, 4. Humanlike, evaluate the following two stories by assigning an integer score (from 1 to 5) to each category. Higher score means better.\nThe initial premise of story is: {}\n\nStory1: {}\n\n Story2: {}.\n\nIn your response, please use the following example format strictly and no need for any extra explanations: \nStory1\nInteresting: 3\nCoherent: 4\nRelevant: 5\nHumanlike: 2\nStory2\nInteresting: 1\nCoherent: 5\nRelevant: 5\nHumanlike: 3\n"

# HANNA_RATE_ESSAY_PROMPT_TEMPLATE="Evaluate the following two stories by assigning an integer score (from 1 to 30) to each. Higher score means better.\nThe initial premise of story is: {}\n\nStory1: {}\n\n Story2: {}.\n\nIn your response, please use the following example format strictly and no need for any extra explanations: \nStory1\nScore: 19\nStory2\nScore: 22\n"


QUARREL_PREMISE = "You will collaborate to create a story. The general setting: A Quarrel between two good friends about Iron Man."
IBRUSIA_PREMISE = "You will collaborate to create a story. The general setting: The state of Ibrusia is coming to a desperate and dangerous situation as the Hosso Union approaches its capital, Zaragoza."
ECONOMY_PREMISE = "You will collaborate to create a story. The general setting: The state of Gurata is coming to a huge economic recession. People are in panic and streets are in turmoil."


def get_response(model, message):
    """
    Query the LLM model with a message and return the response.
    """
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


# baseline_path = 'gemini-ibrusia.txt'
# # generate_baseline('gemini-pro', IBRUSIA_PREMISE, baseline_path)
# for evaluator_model in ['gpt-3.5-turbo-16k', 'gemini-pro', 'anyscale/mistralai/Mistral-7B-Instruct-v0.1']:
#     print(f'Evaluating with {evaluator_model}...')
#     evaluations = evaluate(
#         essay_path='storys/mid_ibrusia.txt',
#         evaluator_model=evaluator_model,
#         premise=IBRUSIA_PREMISE,
#         num_trials=5,
#         baseline_path=baseline_path,
#     )
#     output_path = Path(f'evaluations/{evaluator_model}.txt')
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(output_path, 'w+') as f:
#         f.write(f'********{evaluator_model}********\n')
#         f.write('\n\n\n********************************\n\n\n'.join(evaluations))


def loadCSV(filepath):
    """
    Load a csv file. The first line of the CSV file are the column names.
    Then each subsequent line is a row of data.
    """
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        row_num = 0
        for row in reader:
            if row_num == 0:
                column_names = row
                data = []
            else:
                data.append(row)
            row_num += 1

    return column_names, data

def loadHanna(filepath):
    """
    Load the Hanna dataset. The first line of the CSV file are the column names.
    Then each subsequent line is a row of data.
    """
    num_prompts = 96
    column_names, data = loadCSV(filepath)
    writers = ["Human", "BertGeneration", "CTRL", "GPT", "GPT-2 (tag)", "GPT-2", "RoBERTa", "XLNet", "Fusion", "HINT", "TD-VAE"]

    idx2Prompt = {}
    prompt2Idx = {}
    prompt2Scores = {}
    # Helper dict to ensure each writer version of a story is evaluated three times.
    prompt2AddCount = {}
    prompt2Stories = {}
    idx = 0
    for row in data:
        prompt = row[1].strip()
        if prompt not in prompt2Idx:
            prompt2Idx[prompt] = idx
            idx2Prompt[idx] = prompt
            idx += 1
            prompt2Scores[prompt] = {}
            prompt2AddCount[prompt] = {}
            prompt2Stories[prompt] = {}
            for writer in writers:
                prompt2Scores[prompt][writer] = 0
                prompt2AddCount[prompt][writer] = 0

        writer = row[4].strip()
        story = row[3].strip()

        relevance = int(row[5])
        coherence = int(row[6])
        empathy = int(row[7])
        surprise = int(row[8])
        engagement = int(row[9])
        complexity = int(row[10])
        score = relevance + coherence + empathy + surprise + engagement + complexity


        prompt2Scores[prompt][writer] += score
        prompt2AddCount[prompt][writer] += 1
        prompt2Stories[prompt][writer] = story
    
    for value in prompt2AddCount.values():
        for writer in writers:
            assert value[writer] == 3
    
    assert len(prompt2Idx) == len(idx2Prompt) == len(prompt2Scores) == len(prompt2AddCount) == num_prompts

    return prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories


def splitTrainTest(prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories, num_train):
    """
    Splits the data into train and test sets.
    """
    trainPrompt2Idx = {}
    trainIdx2Prompt = {}
    trainPrompt2Scores = {}
    trainPrompt2Stories = {}

    testPrompt2Idx = {}
    testIdx2Prompt = {}
    testPrompt2Scores = {}
    testPrompt2Stories = {}

    for prompt in prompt2Idx:
        idx = prompt2Idx[prompt]
        if idx < num_train:
            trainPrompt2Idx[prompt] = idx
            trainIdx2Prompt[idx] = prompt
            trainPrompt2Scores[prompt] = prompt2Scores[prompt]
            trainPrompt2Stories[prompt] = prompt2Stories[prompt]
        else:
            testPrompt2Idx[prompt] = idx
            testIdx2Prompt[idx] = prompt
            testPrompt2Scores[prompt] = prompt2Scores[prompt]
            testPrompt2Stories[prompt] = prompt2Stories[prompt]

    assert len(trainPrompt2Idx) == len(trainIdx2Prompt) == len(trainPrompt2Scores) == len(trainPrompt2Stories) == num_train
    assert len(testPrompt2Idx) == len(testIdx2Prompt) == len(testPrompt2Scores) == len(testPrompt2Stories) == len(prompt2Idx) - num_train

    return [trainPrompt2Idx, trainIdx2Prompt, trainPrompt2Scores, trainPrompt2Stories], [testPrompt2Idx, testIdx2Prompt, testPrompt2Scores, testPrompt2Stories]


def extract_first_number(text):
    match = re.search(r'\d+', text)
    if match:
        return int(match.group(0))
    else:
        return None

def parse_scores(response, num_categories=1):
    """
    Parses the scores from the evaluator's response.
    The response should be in the following format (the categories are just examples):
    Story1
    Relevance: 3
    Coherence: 4
    Empathy: 5
    Surprise: 2
    Engagement: 4
    Complexity: 3
    Story2
    Relevance: 3
    Coherence: 4
    Empathy: 5
    Surprise: 2
    Engagement: 4
    Complexity: 3
    """
    try:
        # Splitting the response by newlines
        lines = response.strip().split('\n')
        
        # Placeholder for scores
        llmScore1, llmScore2 = [], []

        # Variable to keep track of which story's scores we are currently parsing
        current_story = 1

        # for line in lines:
        #     if line.startswith("Story1"):
        #         continue
        #     elif line.startswith("Story2"):
        #         current_story = 2
        #     else:
        #         try:
        #             score = int(line.split(': ')[1])
        #         except:
        #             continue
        #         if current_story == 1:
        #             llmScore1.append(score)
        #         else:
        #             llmScore2.append(score)

        for line in lines:
            if line.startswith("Story"):
                story_num = extract_first_number(line)
                if story_num == 1:
                    continue
                elif story_num == 2:
                    current_story = 2
                else:
                    raise ValueError(f"Invalid story number in response: {response}")
            else:
                score = extract_first_number(line)
                if score is None:
                    continue
                if current_story == 1:
                    llmScore1.append(score)
                else:
                    llmScore2.append(score)

        if len(llmScore1) != num_categories or len(llmScore2) != num_categories:
            raise ValueError(f"Incorrect number of scoring categories for one or both stories.\nThe current response is:\n{response}")

        return llmScore1, llmScore2
    
    except Exception as e:
        # Handling any potential errors
        print(f"Error parsing scores: {e}")
        return None, None


def evaluate_stories(model, premises, stories1, stories2, num_categories):
    """
    Using an LLM, evaluates two lists of stories generated by two different entities based on given premises.
    :param model: model to evaluate
    :param premises: list of premises
    :param stories1: list of articles generated by entity1
    :param stories2: list of articles generated by entity2
    :param scores1: list of scores for stories1. Each score is a list of integers for each category.
    :param scores2: list of scores for stories2. Each score is a list of integers for each category.

    :return: list of LLM scores for stories1, list of LLM scores for stories2
    """
    assert len(premises) == len(stories1) == len(stories2)
    
    llmScores1, llmScores2 = [], []
    
    for i in range(len(stories1)):
        premise = premises[i]
        story1 = stories1[i]
        story2 = stories2[i]

        prompt = HANNA_RATE_ESSAY_PROMPT_TEMPLATE.format(premise, story1, story2)
        response = get_response(model, prompt)
        llmScore1, llmScore2 = parse_scores(response, num_categories=num_categories)
        
        if llmScore1 is None or llmScore2 is None:
            raise Exception(f"Error evaluating stories for premise {i+1}:\n\n{premise}")
        else:
            llmScores1.append(llmScore1)
            llmScores2.append(llmScore2)

    llmScores1 = np.array(llmScores1).sum(axis=1)
    llmScores2 = np.array(llmScores2).sum(axis=1)
    return llmScores1, llmScores2


def compute_model_eval_acc(scores1, scores2, llmScores1, llmScores2):
    """
    Computes the model evaluation accuracy given the original and LLM scores.
    """
    assert len(scores1) == len(scores2) == len(llmScores1) == len(llmScores2)

    scores1 = np.array(scores1)
    scores2 = np.array(scores2)
    llmScores1 = np.array(llmScores1)
    llmScores2 = np.array(llmScores2)

    scores1 = scores1.reshape([len(scores1), -1])
    scores2 = scores2.reshape([len(scores2), -1])
    llmScores1 = llmScores1.reshape([len(llmScores1), -1])
    llmScores2 = llmScores2.reshape([len(llmScores2), -1])

    assert len(scores1[0]) == len(scores2[0]) == len(llmScores1[0]) == len(llmScores2[0])

    scores1 = scores1.sum(axis=1)
    scores2 = scores2.sum(axis=1)
    llmScores1 = llmScores1.sum(axis=1)
    llmScores2 = llmScores2.sum(axis=1)

    acc = 0
    
    for i in range(len(scores1)):
        if scores1[i] > scores2[i]:
            if llmScores1[i] > llmScores2[i]:
                acc += 1
        elif scores1[i] < scores2[i]:
            if llmScores1[i] < llmScores2[i]:
                acc += 1
        else:
            if llmScores1[i] == llmScores2[i]:
                acc += 1
        
        # if scores1[i] - scores2[i] >= 9:
        #     if llmScores1[i] > llmScores2[i]:
        #         acc += 1
        # elif scores1[i] - scores2[i] <= -9:
        #     if llmScores1[i] < llmScores2[i]:
        #         acc += 1
        # else:
        #     if np.abs(llmScores1[i] - llmScores2[i]) <= 3:
        #         acc += 1
    return acc / len(scores1)


def evaluateHanna(model, filepath, num_prompts_eval=3, num_categories=1, bidir_eval=False, 
                  eval_rounds=1):
    """
    bidir_eval: whether to evaluate both orders (directions) of presenting the two stories
    eval_rounds: number of rounds of evaluation for the same two stories
    """
    num_train = 48
    assert num_prompts_eval <= num_train

    writers = ["Human", "BertGeneration", "CTRL", "GPT", "GPT-2 (tag)", "GPT-2", "RoBERTa", "XLNet", "Fusion", "HINT", "TD-VAE"]

    prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories = loadHanna(filepath)
    train_set, test_set = splitTrainTest(prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories, num_train)
    trainPrompt2Idx, trainIdx2Prompt, trainPrompt2Scores, trainPrompt2Stories = train_set
    testPrompt2Idx, testIdx2Prompt, testPrompt2Scores, testPrompt2Stories = test_set

    acc = 0
    acc_count = 0
    for i in range(len(writers)):
        writer1 = writers[i]
        for j in range(i+1, len(writers)):
            writer2 = writers[j]
            premises = []
            scores1 = []
            scores2 = []
            stories1 = []
            stories2 = []
            
            for prompt in list(trainPrompt2Idx.keys())[:num_prompts_eval]:
                premises.append(prompt)
                stories1.append(trainPrompt2Stories[prompt][writer1])
                stories2.append(trainPrompt2Stories[prompt][writer2])
                scores1.append(trainPrompt2Scores[prompt][writer1])
                scores2.append(trainPrompt2Scores[prompt][writer2])
            
            llmScores1 = np.zeros(num_prompts_eval)
            llmScores2 = np.zeros(num_prompts_eval)
            for _ in range(eval_rounds):
                tmp_llmScores1, tmp_llmScores2 = evaluate_stories(model, premises, stories1, stories2, num_categories=num_categories)
                llmScores1 += tmp_llmScores1
                llmScores2 += tmp_llmScores2

                # Bidirectional evaluation
                if bidir_eval:
                    tmp_llmScores2, tmp_llmScores1 = evaluate_stories(model, premises, stories2, stories1, num_categories=num_categories)
                    llmScores2 += tmp_llmScores2
                    llmScores1 += tmp_llmScores1

            tmp_acc = compute_model_eval_acc(scores1, scores2, llmScores1, llmScores2)
            acc += tmp_acc
            acc_count += 1
            print(f"Train Accuracy for {writer1} vs {writer2}: {tmp_acc}; cumulative accuracy: {acc / acc_count}")
            
    
    acc /= (len(writers) * (len(writers) - 1) / 2)
    print(f"\nOverall Train Accuracy: {acc}")
    assert acc_count == len(writers) * (len(writers) - 1) / 2


if __name__ == '__main__':
    evaluateHanna('gpt-4-1106-preview', 'hanna/hanna_stories_annotations.csv', num_prompts_eval=2, num_categories=6, bidir_eval=True, eval_rounds=1)
    # evaluateHanna('gpt-3.5-turbo-1106', 'hanna/hanna_stories_annotations.csv', num_prompts_eval=2, num_categories=6, bidir_eval=True, eval_rounds=1)
    # evaluateHanna('gpt-3.5-turbo', 'hanna/hanna_stories_annotations.csv', num_prompts_eval=2, num_categories=6, bidir_eval=True, eval_rounds=1)
