import litellm
import re

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


def get_response(model, message):
    """
    Query the LLM model with a message and return the response.
    """
    response = litellm.completion(
        model=model,
        messages=[{"content": message, "role": "user"}],
    )
    return response.choices[0].message.content


if __name__ == '__main__':
    premise = "A scientific study proves that all humans have been breathing a mind-altering gas from birth. It has been in the air since the beginning of recorded time. People have been in a constant state of being high. Until now. Specialised gas masks are handed out and people have begun to act strange."

    human_story = "When Tyler entered the ward, his daughter Valerie was already fast asleep, her frail body no match for the potent cocktail of drugs coursing through her veins. “She’s been drifting all day, so you didn’t miss much, ” said Roni as she got up to embrace her husband. “How did the appeal go? ” Tyler smiled mournfully. “No luck there. They acknowledged my potential as a Donner, but I failed the psych evaluation again. ‘ Likely to succumb to pressures from family situation’, they said. No matter, we’ll find another way to get the money we need for Valerie’s treatments.” Tyler and Roni sat quietly as they cast furtive glances at Valerie, each lost in their own thoughts. Roni was the first to break the silence. “Did they say which project you would have been assigned to if you became a Donner? ” “They did, in fact. There’s an opening on the Renewable Energies team. They think they’re on the verge of a breakthrough, and one additional Donner is all they need to beat the Chinese competitors to the patenting deadline.” Tyler sighed, then leaned back in his chair and closed his eyes before continuing, “The hazard pay was really good too. 5 years’salary for just one month of being a Donner! And full psych after-care thrown in as well! ” Roni’s grip on Tyler’s hand tightened then. “Have the suicide rates… decreased with the psych after-care? ” “That’s what they claim, at least. Some still believe that the utter desolation one experiences with the absence of Perogon-X2 is irreversible, but hey, if that’s the price for increased mental faculties, up to a 100-point increase in IQ, there’ll always be people willing to pay.” Roni fished out a print-out from her handbag, and passed it over to Tyler. “This other group believes that Perogon-X2 is a good rather than bad thing. This ‘ naturally-occurring high’, as they call it, is the only thing keeping us from falling into a spiralling abyss of depression ... it’s the proverbial wool over our eyes, but for our own good.” “You and I are both scientists, ” laughed Tyler, “but you remember how we both thought that this surely was proof of some higher power too when the discovery was first publicised? Two birds with one stone! It keeps the majority of us merrily alive, while also ensuring that we would not run about unlocking the secrets of the universe until we were ready? ” There was no denying the contributions Donners had made to humankind. Tyler casually glanced around the room and out the window, and easily counted a dozen inventions which could not have been possible without the Donners. The hovercars, the bacterial foods with customizable tastes, implantable microchips for constant connection to the internet… even the drugs which commuted what would have been a death sentence for Valerie a decade ago to a mere ( if expensive ) annoyance. In a way, Tyler felt relief from having failed to qualify as a Donner. He had seen first-hand how some of the most emotionally-staunch people had been reduced to forlorn desolate souls once the Perogon-X2 was filtered out with the Masks. The degenerative process was universal – Donners would immediately benefit from heightened intelligence, but over time, they would latch onto and stubbornly nurse the perception that life was utterly… meaningless. Even if Donners were subsequently re-exposed to Perogon-X2, only 2 in 10 ever adapted back to normal life. The rest… preferred to simply put an end to it all. Roni’s voice snapped Tyler out of his reverie. “Tyler, remember how I begged you not to apply to be a Donner? ” “Yes, I remember.” “And you told me that you had no choice, because Valerie meant the world to you, and this was the least you could do as a parent for her? ” “Why are you bringing that up now? ” said Tyler with a flash of irritation. “I told you, I’ve been rejected. I can’t be a Donner anymore. We’ll just have to find another way.” Roni reached back into her handbag, and passed over a crisply-folded letter on embossed paper. Tyler began reading, and while he remained expressionless, Roni noted the bulging vein in the middle of his forehead, Tyler’s tell-tale indication of agitation. Roni wondered how much resistance Tyler was going to muster, but in her heart, she knew what she had to do. “I applied as well, Tyler, and they accepted me. They’ve already fitted my Mask, and I’ll be Donning it next Monday. We’ll have enough for Valerie after all.” -- - /r/rarelyfunny"

    model_story = "I just want to tell you to go home. You will be fine. We're doing your next five minutes to reach the next building and feed on that mess that is the past you have made and you will lead a healthy life and that is something that I will teach you to have. Thanks for reading! ⁇ Have you ever tried r/WritingPrompts? More than ever you come up with a series of prompts that inspire you to write about the prompt, or the world you post. I don't do magic, but I write when I write about it. For example, to your friends of mine, I read a prompt that got a writing prompt and got asked to post it and my comment worked. No, I write about plants or science, and that answer is wrong. The ground is bigger than the sky, the elements in the universe are different, where you put you feet together, how you feel and what you taste, who you are, what you want, who you look like, etc etc etc etc etc etc. And it's not much of a *leoderie* and I can't say that it's a lovely thing, but I hope you like it, it's an enjoyable story. EDIT: A word word. ⁇"

    PROMPT_TEMPLATE_1="Based on the following six attributes:\n1. Relevance (how well the story matches its prompt).\n2. Coherence (how much the story makes sense).\n3. Empathy (how well the reader can understand the character’s emotions).\n4. Surprise (how surprising the end of the story is).\n5. Engagement (how much the reader can engage with the story).\n6. Complexity (how elaborate the story is).\nGive an integer rating (from 1 to 5, higher means better) to each attribute of the following story first, and then give the explanation of the ratings.\n\nThe initial prompt: {}\n\nThe story: {}\n\nIn your response, please use the following format:\n###Ratings###\n*Relevance*: \n*Coherence*: \n*Empathy*: \n*Surprise*: \n*Engagement*: \n*Complexity*: \n\n###Explanations###:"

    PROMPT_TEMPLATE_2 = "Based on the following six categories:\n1. Relevance (how well the story matches its prompt).\n2. Coherence (how much the story makes sense).\n3. Empathy (how well the reader can understand the character’s emotions).\n4. Surprise (how surprising the end of the story is).\n5. Engagement (how much the reader can engage with the story).\n6. Complexity (how elaborate the story is).\nEvaluate the following two stories by assigning an integer score (from 1 to 5) to each category. Higher score means better.\n\nThe initial premise of story is: {}\n\nStory1:\n{}\n\nStory2:\n{}.\n\nIn your response, please use the following format: \n###Story1 Ratings###\n*Relevance*: \n*Coherence*: \n*Empathy*: \n*Surprise*: \n*Engagement*: \n*Complexity*: \n\n###Story2 Ratings###\n*Relevance*: \n*Coherence*: \n*Empathy*: \n*Surprise*: \n*Engagement*: \n*Complexity*: \n\n###Explanations###:"


    prompt = PROMPT_TEMPLATE_2.format(premise, human_story, model_story)
    print(prompt)
    print("=====================================================")
    response = get_response('gpt-3.5-turbo-0125', prompt)
    # response = get_response('gpt-4-0125-preview', prompt)
    # response = get_response('gemini-pro', prompt)
    print(response)

    # s = "*Relevance*: "
    # print(extract_first_number(s))