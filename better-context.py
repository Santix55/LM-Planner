from unified_planning.shortcuts import *
from unified_planning.io import PDDLReader

from transformers import AutoTokenizer, AutoModelForCausalLM

from auxilar.verbose_print import verbose_print_factory

from textwrap import dedent
import re
import json

from collections import deque


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
        1. "analysis": Here you can reason step by step, all that you want to choose the next action. Check the action history to **AVOID** loops.
        2. "action": **ONLY THE FIRST** PDDL-formatted action
        3. "index": The index of the actions in the users list
        
        Strict rules:
        - Use only standard ASCII characters
        - Include no text outside the JSON
        - Keys must be lowercase and double-quoted
        - Action must match exact PDDL syntax from available actions
        - Never choose an action outside of the list
    """) + str(problem)
    sysprompt_tokens = len(tokenizer.tokenize(system_prompt)); vprint(f"# system prompt tokens = {sysprompt_tokens}")
    
    # Inicializar ventana deslizante de contexto
    MAX_WIN_SIZE = tokenizer.model_max_length-100-sysprompt_tokens;   vprint(f"# MAX WIN SIZE= {MAX_WIN_SIZE}")
    context = deque([]); win_size = 0; total_deleted = 0

    ## BUCLE DE BÚSQUEDA ##
    plan = []; goal = False; 
    for n_steps in range(max_steps):
        vprint(f"\n\n== STEP {n_steps} ==")

        # [i] Mostrar información sobre la ventana de contexto
        vprint(f"# window size = {win_size/MAX_WIN_SIZE}")
        vprint(f"# acciones borradas = {total_deleted}")
        
        # Context Prompt (mostrar las acciones pasadas)
        context_prompt = "This is the list of your previous actions: \n"
        context_prompt += "\n".join([generation[0] for generation in context]) + "\n\n"
        if len(context) == 0: context_prompt = ""

        # User Prompt (historial de acciones + estado actual + lista the acciones aplicables)
        user_prompt = f"This is the current state of the enviorment:\n {current_state} \n"
        for statement in current_state._values:
            user_prompt += str(statement)
        user_prompt += "\n\n"
        
        user_prompt += f"This is the list of your previous actions: \n"
        for i, action_step in enumerate(plan):
            user_prompt += f"{i}.-{action_step}\n"
        user_prompt += "\n"

        user_prompt += "The list of posible actions for this state is: \n"
        applicable_actions = list(simulator.get_applicable_actions(current_state))
        for i, applicable_action in enumerate(applicable_actions):
            user_prompt += f"{i}.-{repr_applicable_action(applicable_action)}\n"

        # Construir el prompt entero para que lo complete
        full_prompt = f"<|system|>\n{system_prompt}\n<|end|>\n\n<|user|>\n{context_prompt}{user_prompt}\n<|end|>\n<|assistant|>\n"
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            **generation_config
        )
        input_tokens = inputs.input_ids.shape[1];   #vprint(f"# input tokens = {input_tokens}")
        response_tokens = outputs.shape[1];         #vprint(f"# response tokens = {response_tokens}")
        new_tokens = response_tokens-input_tokens;  vprint(f"# new tokens = {new_tokens}")
        response = tokenizer.decode(outputs[0], skip_special_tokens=False); vprint(response)

        # Extraer la última respuesta como solución
        pattern = r"<\|assistant\|>(.*?)<\|end\|>"
        answer_str = re.findall(pattern, response, re.DOTALL)[-1].strip()
        answer_str = re.sub(r"```json\n?", "", answer_str)  # Elimina apertura
        answer_str = re.sub(r"\n?```$", "", answer_str)     # Elimina cierre
        answer_str = answer_str.strip()                     # Limpia espacios
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
        
        # [+] Añadir al contexto la última acción realizada
        context.append((answer_str, new_tokens))
        win_size += new_tokens

        # [-] Ir eliminando acciones del contexto hasta que no sobrepase el limite de la ventana
        n_deleted = 0
        while win_size > MAX_WIN_SIZE:
            win_size -= context.popleft()[1]
            n_deleted += 1
        total_deleted += n_deleted


    
    print("\n\n\n=== PLAN OBTENIDO ===")
    print(f"> pasos: {len(plan)}")
    print(f"> completado: {goal}")
    for action_step in plan:
        print(action_step)

if __name__ == "__main__":
    path_domain  = "./pddl/domain-03.pddl"
    path_problem = "./pddl/problem-03.pddl"

    verbose = True

    max_steps = 20

    main(path_domain=path_domain, path_problem=path_problem, max_steps=max_steps, verbose=verbose)
