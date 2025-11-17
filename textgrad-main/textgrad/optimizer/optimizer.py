from abc import ABC, abstractmethod
from typing import List, Union
from collections import defaultdict

import base64
import binascii
from textgrad.variable import Variable
from textgrad import logger
from textgrad.engine import EngineLM, validate_multimodal_engine
from textgrad.config import validate_engine_or_get_default
from .optimizer_prompts import (
    construct_image_optimization_prompt,
    construct_tgd_prompt,
    IMAGE_OPTIMIZER_SYSTEM_PROMPT,
    OPTIMIZER_SYSTEM_PROMPT,
    GRADIENT_TEMPLATE,
    GRADIENT_MULTIPART_TEMPLATE,
)


def get_gradient_and_context_text(variable) -> Union[str, List[Union[str, bytes]]]:
    """For the variable, aggregates and returns 
    i. the gradients 
    ii. the context for which the gradients are computed.

    This is used by the optimizer.  
    :return: A string containing the aggregated gradients and their corresponding context.
    :rtype: str
    """

    gradient_content = []
    for g in variable.gradients:
        if variable.gradients_context[g] is None:
            gradient_content.append(g.value)
        else:
            # If context is a list, we handle it differently.
            context = variable.gradients_context[g]
            if isinstance(context["context"], str):
                # The context could be all string.
                criticism_and_context = GRADIENT_TEMPLATE.format(
                    feedback=g.value, **context)
                gradient_content.append(criticism_and_context)
            elif isinstance(context["context"], list):
                # The context may have a list of images / strings. In this case, we need to handle it differently.
                context_prompt = GRADIENT_MULTIPART_TEMPLATE.format(**context, feedback=g.value)
                criticism_and_context = context["context"] + [context_prompt]
                gradient_content.extend(criticism_and_context)
            else:
                raise ValueError("Context must be either a string or a list.")
    
    # Check if all instances are string
    if all(isinstance(i, str) for i in gradient_content):
        return "\n".join(gradient_content)
    else:
        return gradient_content


class Optimizer(ABC):
    """
    Base class for all optimizers.

    :param parameters: The list of parameters to optimize.
    :type parameters: List[Variable]

    :Methods:
        - zero_grad(): Clears the gradients of all parameters.
        - step(): Performs a single optimization step.
    """

    def __init__(self, parameters: List[Variable]):
        
        self.parameters = parameters
        
    def zero_grad(self):
        """
        Clears the gradients of all parameters.
        """
        for p in self.parameters:
            p.gradients = set()

    @abstractmethod
    def step(self):
        """
        Performs a single optimization step.
        """
        pass


class TextualGradientDescent(Optimizer):
    def __init__(self, 
                 parameters: List[Variable], 
                 verbose: int=0, 
                 engine: Union[EngineLM, str]=None, 
                 constraints: List[str]=None,
                 new_variable_tags: List[str]=None,
                 optimizer_system_prompt: str=OPTIMIZER_SYSTEM_PROMPT,
                 in_context_examples: List[str]=None,
                 gradient_memory: int=0):
        """TextualGradientDescent optimizer

        :param engine: the engine to use for updating variables
        :type engine: EngineLM
        :param parameters: the parameters to optimize
        :type parameters: List[Variable]
        :param verbose: whether to print iterations, defaults to 0
        :type verbose: int, optional
        :param constraints: a list of natural language constraints, defaults to []
        :type constraints: List[str], optional
        :param optimizer_system_prompt: system prompt to the optimizer, defaults to textgrad.prompts.OPTIMIZER_SYSTEM_PROMPT. Needs to accept new_variable_start_tag and new_variable_end_tag
        :type optimizer_system_prompt: str, optional
        :param in_context_examples: a list of in-context examples, defaults to []
        :type in_context_examples: List[str], optional
        :param gradient_memory: the number of past gradients to store, defaults to 0
        :type gradient_memory: int, optional
        """
        super().__init__(parameters)

        for parameter in self.parameters:
            if not isinstance(parameter.value, str):
                raise NotImplementedError(
                    "TextualGradientDescent only supports string variables. "
                    "Use MultimodalGradientDescent for image parameters."
                )

        if new_variable_tags is None:
            new_variable_tags = ["<IMPROVED_VARIABLE>", "</IMPROVED_VARIABLE>"]

        self.engine = validate_engine_or_get_default(engine)
        self.verbose = verbose
        self.constraints = constraints if constraints is not None else []
        self.optimizer_system_prompt = optimizer_system_prompt.format(new_variable_start_tag=new_variable_tags[0], new_variable_end_tag=new_variable_tags[1])
        self.do_constrained = (len(self.constraints) > 0)
        self.new_variable_tags = new_variable_tags
        self.in_context_examples = in_context_examples if in_context_examples is not None else []
        self.do_in_context_examples = (len(self.in_context_examples) > 0)
        self.gradient_memory = gradient_memory
        self.gradient_memory_dict = defaultdict(list)
        self.do_gradient_memory = (gradient_memory > 0)

    @property
    def constraint_text(self):
        """
        Returns a formatted string representation of the constraints.

        :return: A string containing the constraints in the format "Constraint {index}: {constraint}".
        :rtype: str
        """
        constraints_ordered = [f"Constraint {i+1}: {constraint}" for i, constraint in enumerate(self.constraints)]
        return "\n".join(constraints_ordered)
    
    def get_gradient_memory_text(self, variable: Variable):
        grad_memory = ""
        variable_grad_memory = self.gradient_memory_dict[variable][-self.gradient_memory:]
        for i, grad_info in enumerate(variable_grad_memory):
            grad_memory += f"\n<FEEDBACK-{i+1}> {grad_info['value']}</FEEDBACK-{i+1}>\n"
        return grad_memory
    
    def update_gradient_memory(self, variable: Variable):
        self.gradient_memory_dict[variable].append({"value": variable.get_gradient_text()})
    
    def _update_prompt(self, variable: Variable) -> Union[str, List[Union[str, bytes]]]:
        grad_memory = self.get_gradient_memory_text(variable)
        optimizer_information = {
            "variable_desc": variable.get_role_description(),
            "variable_value": variable.value,
            "variable_grad": get_gradient_and_context_text(variable),
            "variable_short": variable.get_short_value(),
            "constraint_text": self.constraint_text,
            "new_variable_start_tag": self.new_variable_tags[0],
            "new_variable_end_tag": self.new_variable_tags[1],
            "in_context_examples": "\n".join(self.in_context_examples),
            "gradient_memory": grad_memory,
            "past_values": grad_memory,
        }
        
        prompt = construct_tgd_prompt(do_constrained=self.do_constrained, 
                                      do_in_context_examples=(self.do_in_context_examples and (len(self.in_context_examples) > 0)),
                                      do_gradient_memory=(self.do_gradient_memory and (grad_memory != "")),
                                      **optimizer_information)
        
        logger.info(f"TextualGradientDescent prompt for update", extra={"prompt": prompt})
        return prompt

    def step(self, **sample_params):
        """
        Perform a single optimization step.
        This method updates the parameters of the optimizer by generating new text using the engine and updating the parameter values accordingly.
        It also logs the optimizer response and the updated text.
        Returns:
            None
        """
        for parameter in self.parameters:
            prompt_update_parameter = self._update_prompt(parameter)
            new_text = self.engine(prompt_update_parameter, system_prompt=self.optimizer_system_prompt, **sample_params)

            if len(new_text) == 1:
                logger.info(f"TextualGradientDescent optimizer response", extra={"optimizer.response": new_text})
                try:
                    new_value = new_text.split(self.new_variable_tags[0])[1].split(self.new_variable_tags[1])[0].strip()
                # Check if we got a cannot be indexed error
                except IndexError:
                    logger.error(f"TextualGradientDescent optimizer response could not be indexed", extra={"optimizer.response": new_text})
                    raise IndexError(f"TextualGradientDescent optimizer response could not be indexed. This can happen if the optimizer model cannot follow the instructions. You can try using a stronger model, or somehow reducing the context of the optimization. Response: {new_text}")
                parameter.set_value(new_value)
                logger.info(f"TextualGradientDescent updated text", extra={"parameter.value": parameter.value})
                if self.verbose:
                    print("-----------------------TextualGradientDescent------------------------")
                    print(parameter.value)
                
                if self.do_gradient_memory:
                    self.update_gradient_memory(parameter)
                return [new_text]
            else: # multiple samples
                extracted_texts = []
                for _text in new_text:
                    try:
                        extracted_texts.append(_text.split(self.new_variable_tags[0])[1].split(self.new_variable_tags[1])[0].strip())
                    except:
                        continue


                return extracted_texts

class MultimodalGradientDescent(Optimizer):
    def __init__(self,
                 parameters: List[Variable],
                 engine: Union[EngineLM, str] = None,
                 constraints: List[str] = None,
                 new_variable_tags: List[str] = None,
                 optimizer_system_prompt: str = IMAGE_OPTIMIZER_SYSTEM_PROMPT,
                 in_context_examples: List[str] = None,
                 gradient_memory: int = 0,
                 verbose: int = 0):
        super().__init__(parameters)

        for parameter in self.parameters:
            if not isinstance(parameter.value, (bytes, bytearray)):
                raise NotImplementedError(
                    "MultimodalGradientDescent expects image variables encoded as bytes."
                )

        if new_variable_tags is None:
            new_variable_tags = ["<IMPROVED_VARIABLE>", "</IMPROVED_VARIABLE>"]

        self.engine = validate_engine_or_get_default(engine)
        validate_multimodal_engine(self.engine)

        self.verbose = verbose
        self.constraints = constraints if constraints is not None else []
        self.optimizer_system_prompt = optimizer_system_prompt.format(
            new_variable_start_tag=new_variable_tags[0],
            new_variable_end_tag=new_variable_tags[1],
        )
        self.do_constrained = len(self.constraints) > 0
        self.new_variable_tags = new_variable_tags
        self.in_context_examples = in_context_examples if in_context_examples is not None else []
        self.do_in_context_examples = len(self.in_context_examples) > 0
        self.gradient_memory = gradient_memory
        self.do_gradient_memory = gradient_memory > 0
        self.gradient_memory_dict = defaultdict(list)

    @property
    def constraint_text(self) -> str:
        constraints_ordered = [f"Constraint {i+1}: {constraint}" for i, constraint in enumerate(self.constraints)]
        return "\n".join(constraints_ordered)

    def get_gradient_memory_text(self, variable: Variable) -> str:
        grad_memory = ""
        variable_grad_memory = self.gradient_memory_dict[variable][-self.gradient_memory:]
        for i, grad_info in enumerate(variable_grad_memory):
            grad_memory += f"\n<FEEDBACK-{i+1}> {grad_info['value']}</FEEDBACK-{i+1}>\n"
        return grad_memory

    def update_gradient_memory(self, variable: Variable):
        self.gradient_memory_dict[variable].append({"value": variable.get_gradient_text()})

    def _update_prompt(self, variable: Variable) -> List[Union[str, bytes]]:
        grad_memory = self.get_gradient_memory_text(variable)
        gradients = get_gradient_and_context_text(variable)
        if isinstance(gradients, list):
            context_segments = list(gradients)
            gradient_summary = "Refer to the preceding multimodal conversation and feedback segments."
        else:
            context_segments = []
            gradient_summary = gradients

        optimizer_information = {
            "variable_desc": variable.get_role_description(),
            "variable_grad": gradient_summary,
            "constraint_text": self.constraint_text,
            "new_variable_start_tag": self.new_variable_tags[0],
            "new_variable_end_tag": self.new_variable_tags[1],
            "in_context_examples": "\n".join(self.in_context_examples),
            "gradient_memory": grad_memory,
            "past_values": grad_memory,
        }

        prompt_tail = construct_image_optimization_prompt(
            do_constrained=self.do_constrained,
            do_in_context_examples=(self.do_in_context_examples and (len(self.in_context_examples) > 0)),
            do_gradient_memory=(self.do_gradient_memory and (grad_memory != "")),
            **optimizer_information,
        )

        prompt_segments: List[Union[str, bytes]] = [bytes(variable.get_value())]
        if context_segments:
            prompt_segments.extend(context_segments)
        prompt_segments.append(prompt_tail)
        return prompt_segments

    def _decode_response(self, response: str) -> bytes:
        try:
            payload = response.split(self.new_variable_tags[0])[1].split(self.new_variable_tags[1])[0].strip()
        except IndexError as exc:
            raise ValueError(
                "MultimodalGradientDescent optimizer response missing improved image tags."
            ) from exc

        try:
            return base64.b64decode(payload)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Failed to decode improved image payload from optimizer response.") from exc

    def step(self, **sample_params):
        for parameter in self.parameters:
            prompt_update_parameter = self._update_prompt(parameter)
            raw_response = self.engine(
                prompt_update_parameter,
                system_prompt=self.optimizer_system_prompt,
                **sample_params,
            )

            responses = raw_response if isinstance(raw_response, list) else [raw_response]
            decoded_images = []
            for response in responses:
                try:
                    decoded_images.append(self._decode_response(response))
                except ValueError as exc:
                    logger.error(
                        "MultimodalGradientDescent failed to parse optimizer response.",
                        extra={"optimizer.response": response, "error": str(exc)},
                    )
                    continue

            if not decoded_images:
                raise ValueError("MultimodalGradientDescent could not decode any improved images.")

            parameter.set_value(decoded_images[0])
            logger.info(
                "MultimodalGradientDescent updated image",
                extra={"parameter.summary": parameter.get_short_value()},
            )

            if self.verbose:
                print("---------------------MultimodalGradientDescent----------------------")
                print(parameter.get_short_value())

            if self.do_gradient_memory:
                self.update_gradient_memory(parameter)

            if len(decoded_images) == 1:
                return decoded_images[0]

            return decoded_images


class TextualGradientDescentwithMomentum(Optimizer):
    def __init__(self, 
                 engine: Union[str, EngineLM], 
                 parameters: List[Variable], 
                 momentum_window: int = 0, 
                 constraints: List[str]=None,
                 new_variable_tags: List[str]=None,
                 in_context_examples: List[str]=None,
                 optimizer_system_prompt: str=OPTIMIZER_SYSTEM_PROMPT):
        super().__init__(parameters)

        for parameter in self.parameters:
            if not isinstance(parameter.value, str):
                raise NotImplementedError(
                    "TextualGradientDescentwithMomentum only supports string variables."
                )

        if new_variable_tags is None:
            new_variable_tags = ["<IMPROVED_VARIABLE>", "</IMPROVED_VARIABLE>"]

        self.engine = validate_engine_or_get_default(engine)
        
        if momentum_window == 0:
            return TextualGradientDescent(engine=engine, parameters=parameters, constraints=constraints)

        # Each item in the momentum storage will include past value and the criticism
        self.momentum_storage = [[] for _ in range(len(parameters))]
        self.momentum_window = momentum_window
        self.do_momentum = True
        self.constraints = constraints if constraints is not None else []
        self.do_constrained = (len(self.constraints) > 0)
        self.optimizer_system_prompt = optimizer_system_prompt.format(new_variable_start_tag=new_variable_tags[0], new_variable_end_tag=new_variable_tags[1])
        self.new_variable_tags = new_variable_tags
        self.in_context_examples = in_context_examples if in_context_examples is not None else []
        self.do_in_context_examples = (len(self.in_context_examples) > 0)

        logger.info(f"TextualGradientDescent initialized with momentum window: {momentum_window}")

    @property
    def constraint_text(self):
        constraints_ordered = [f"Constraint {i+1}: {constraint}" for i, constraint in enumerate(self.constraints)]
        return "\n".join(constraints_ordered)
    
    def _update_prompt(self, variable: Variable, momentum_storage_idx: int):
        past_values = ""
        
        past_n_steps = self.momentum_storage[momentum_storage_idx]
        for i, step_info in enumerate(past_n_steps):
            past_values += f"\n{variable.get_role_description()} at Step {i + 1}: {step_info['value']}.\n"

        optimizer_information = {
            "variable_desc": variable.get_role_description(),
            "variable_value": variable.value,
            "variable_grad": variable.get_gradient_text(),
            "variable_short": variable.get_short_value(),
            "constraint_text": self.constraint_text,
            "past_values": past_values,
            "new_variable_start_tag": self.new_variable_tags[0],
            "new_variable_end_tag": self.new_variable_tags[1],
            "in_context_examples": "\n".join(self.in_context_examples)
        }
        
        prompt = construct_tgd_prompt(do_momentum=(self.do_momentum and (past_values != "")), 
                                      do_constrained=self.do_constrained, 
                                      do_in_context_examples=(self.do_in_context_examples and (len(self.in_context_examples) > 0)),
                                      **optimizer_information)
        
        logger.info(f"TextualGradientwithMomentum prompt for update", extra={"prompt": prompt})


    def _update_momentum_storage(self, variable: Variable, momentum_storage_idx: int):
        if len(self.momentum_storage[momentum_storage_idx]) >= self.momentum_window:
            self.momentum_storage[momentum_storage_idx].pop(0)
        
        self.momentum_storage[momentum_storage_idx].append({"value": variable.value, "gradients": get_gradient_and_context_text(variable)})
        
    def step(self):
        for idx, parameter in enumerate(self.parameters):
            self._update_momentum_storage(parameter, momentum_storage_idx=idx)
            prompt_update_parameter = self._update_prompt(parameter, momentum_storage_idx=idx)
            new_text = self.engine(prompt_update_parameter, system_prompt=self.optimizer_system_prompt)
            logger.info(f"TextualGradientDescentwithMomentum optimizer response", extra={"optimizer.response": new_text})
            try:
                new_value = new_text.split(self.new_variable_tags[0])[1].split(self.new_variable_tags[1])[0].strip()
            # Check if we got a cannot be indexed error
            except IndexError:
                logger.error(f"TextualGradientDescent optimizer response could not be indexed", extra={"optimizer.response": new_text})
                raise IndexError(f"TextualGradientDescent optimizer response could not be indexed. This can happen if the optimizer model cannot follow the instructions. You can try using a stronger model, or somehow reducing the context of the optimization. Response: {new_text}")
            parameter.set_value(new_value)
            logger.info(f"TextualGradientDescentwithMomentum updated text", extra={"parameter.value": parameter.value})

