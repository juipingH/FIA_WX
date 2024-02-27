from ibm_watson_machine_learning.foundation_models import Model

from core import config
from .prompt_store import generate_main_prompt, generate_translation_prompt, evaluate_question_prompt

model_id = config.answer_generation_model_name
parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 1800,
    "stop_sequences": ["<<完>>"],
    'temperature': 0.1,
}

eval_parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 20,
    "stop_sequences": ["\n"],}


model = Model(
	model_id = model_id,
	params = parameters,
	credentials = config.creds,
	project_id = config.project_id,
	)
eval_model = Model(
	model_id = model_id,
	params = parameters,
	credentials = config.creds,
	project_id = config.project_id,
	)

def evaluate_question(question):
    evaluation_prompt = evaluate_question_prompt(question)
    eval_question = eval_model.generate_text(evaluation_prompt)
    print(eval_question)
    return eval_question

def generate_answer(question, df, chunks):

    prompts = generate_main_prompt(question, chunks, df)
    answer =  model.generate_text(prompts)
    return answer.replace("\\n\\n","\n")

def translate_answer(answer):
    translation_prompt = generate_translation_prompt(answer)
    return model.generate_text(translation_prompt)
