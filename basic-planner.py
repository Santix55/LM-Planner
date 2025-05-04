from unified_planning.shortcuts import *
from unified_planning.io import PDDLReader

from transformers import AutoTokenizer, AutoModelForCausalLM

from textwrap import dedent

from auxilar.verbose_print import verbose_print_factory

import re
import json

def repr_applicable_action(applicable_action):
    action = applicable_action[0].name
    params = applicable_action[1]
    params_repr = [repr(param) for param in params]
    return f"{action}({', '.join(params_repr)})"

def main(path_domain, path_problem, max_steps, verbose):
    vprint = verbose_print_factory(verbose)

    # Iniciar el entorno de simulación
    pddlReader = PDDLReader()
    problem = pddlReader.parse_problem(path_domain, path_problem)
    simulator = SequentialSimulator(problem)
    current_state = simulator.get_initial_state()

    # Configurar el modelo de lenguaje
    model_name = "microsoft/Phi-4-mini-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto"           # distribuye el modelo automáticamente en dispositivo(s) disponibles.
    )
    generation_config = {
        "max_new_tokens": 1000,    # Aumentar espacio para respuesta completa
        "do_sample": True,         # Activar muestreo controlado
        "temperature": 0.3,        # Balance coherencia/exploración
        "top_p": 0.9,              # Nucleus sampling
        "pad_token_id": tokenizer.eos_token_id
    }

    # System Prompt
    system_prompt = dedent("""\
        You are a PPDL planner. Respond **EXCLUSIVELY** with a valid JSON containing:
        1. "analysis": 1-line explanation of your reasoning
        2. "action": **ONLY THE FIRST** PDDL-formatted action
        3. "index": The index of the actions in the users list

        Strict rules:
        - Use only standard ASCII characters
        - Include no text outside the JSON
        - Keys must be lowercase and double-quoted
        - Action must match exact PDDL syntax from available actions
    """) + str(problem)

    plan = []; goal = False
    for n_steps in range(max_steps):
        vprint(f"\n\n== STEP {n_steps} ==")

        # User Prompt (contexto + lista the acciones aplicables)
        user_prompt = f"This is the current state of the enviorment:\n {current_state} \n\n"
        user_prompt += "The list of posible actions for this state is: \n"
        applicable_actions = list(simulator.get_applicable_actions(current_state))
        for i, applicable_action in enumerate(applicable_actions):
            user_prompt += f"{i}.-{repr_applicable_action(applicable_action)}\n"

        # Construir el prompt entero para que lo complete
        full_prompt = f"<|system|>\n{system_prompt}\n<|end|>\n<|user|>\n{user_prompt}\n<|end|>\n<|assistant|>\n"
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            **generation_config
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        vprint(response)

        # Extraer la última respuesta como solución
        pattern = r"<\|assistant\|>(.*?)<\|end\|>"
        answer_str = re.findall(pattern, response, re.DOTALL)[-1].strip()
        answer_dict = json.loads(answer_str)
        
        # Aplicar la acción que te propone el sistema
        index = answer_dict["index"]
        chosen_action, chosen_params = applicable_actions[index]
        current_state = simulator.apply(current_state, chosen_action, chosen_params)

        # Añadir la acción escogida al plan
        plan.append(repr_applicable_action(applicable_actions[index]))

        # [X] Comprobar si se ha llegado a la meta
        if simulator.is_goal(current_state):
            goal = True; break
    
    print("\n\n\n=== PLAN OBTENIDO ===")
    print(f"> pasos: {len(plan)}")
    print(f"> completado: {goal}")
    for action_step in plan:
        print(action_step)

if __name__ == "__main__":
    path_domain  = "./pddl/domain-03.pddl"
    path_problem = "./pddl/problem-03.pddl"

    verbose = True

    max_steps = 7
    main(path_domain=path_domain, path_problem=path_problem, max_steps=max_steps, verbose=verbose)