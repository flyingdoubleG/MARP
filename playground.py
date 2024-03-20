import litellm
import re
from dataset_loader import *

from standard_prompts import *

litellm.vertex_project = "multi-agent-411823"
litellm.vertex_location = "us-central1"

# Helper function
def extract_first_number(text):
        """
        Helper function to extract the first number encountered within a string.
        """
        match = re.search(r'\d+', text)
        if match:
            return int(match.group(0))
        else:
            return None


def get_response(model, message, temperature=None, top_p=None):
    """
    Query the LLM model with a message and return the response.
    """
    response = litellm.completion(
        model=model,
        messages=[{"content": message, "role": "user"}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=4096,
    )
    return response.choices[0].message.content


if __name__ == '__main__':
    # prompt = PROMPT_TEMPLATE_3.format(premise, human_story, gpt2_story)
    # prompt = PROMPT_TEMPLATE_2.format(premise, gpt2_story, gpt2_tag_story)
    # print(prompt)
    # print("=====================================================")
    # response = get_response('gemini-pro', prompt)
    # response = get_response('gpt-4-0125-preview', prompt)
    # response = get_response('gemini-pro', prompt)
    # print(response)

    # s = "*Relevance*: "
    # print(extract_first_number(s))

    writers = ["LEAD-3", "NEUSUM", "BanditSum", "RNES", "Point Generator", "Fast-abs-rl", "Bottom-Up", "Improve-abs", "Unified-ext-abs", "ROUGESal", "Multi-task", "Closed book decoder", "T5", "GPT-2", "BART", "Pegasus"]

    path = "SummEval/model_annotations.aligned.paired.jsonl"
    loader = DatasetLoader("SummEval", path, writers)
    prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories = loader.process_data()
    print(len(prompt2Idx), len(idx2Prompt), len(prompt2Scores), len(prompt2Stories))

    input = idx2Prompt[95]
    output1 = prompt2Stories[input]['T5']
    output2 = prompt2Stories[input]['BART']

    print("\nInput:\n")
    print(input)
    print("\nOutput1:\n")
    print(output1)
    print("\nOutput2:\n")
    print(output2)

    prompt = SUMMEVAL_RATE_DOUBLE_ESSAY_PROMPT_TEMPLATE.format(input, output2, output1)
    # prompt = SUMMEVAL_ANALYZE_RATE_DOUBLE_ESSAY_PROMPT_TEMPLATE.format(input, output1, output2)
    # prompt = SUMMEVAL_ANALYZE_RATE_DOUBLE_ESSAY_PROMPT_TEMPLATE.format(input, output2, output1)
    print("=====================================================")
    print(prompt)
    print("=====================================================")
    # model_full_name = "gemini-pro"
    # model_full_name = "claude-3-haiku-20240307"
    model_full_name = "gpt-3.5-turbo-0125"
    response = get_response(model_full_name, prompt)
    print(response)