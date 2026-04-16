"""
Owner: isst
Modified for Gemma4 two-step T2I prompt generation.
Adds a second vLLM engine call between preprocess and postprocess.
"""
import json
import os
import time
import utils
from llm_opt.oaas_wrapper_v2 import OaasWrapper

import sys
sys.path.append('/Model')
from dlis_inter import PreAndPostProcessor


class ModelImp:
    def  __init__(self):
        self.model_path = utils.get_model_path()
        self.model_dir = os.path.join(os.path.dirname(os.path.realpath(self.model_path)), "model")

        self.data_path = utils.get_data_path()
        self.data_dir = None
        if self.data_path is not None:
            self.data_dir = os.path.dirname(os.path.realpath(self.data_path))
        self.initial_dyanmic_data_paths = utils.get_initial_dynamic_data_paths()
        self.initial_dynamic_data_dirs = utils.get_named_directories(self.initial_dyanmic_data_paths)

        print("Model Path: {model_path}".format(model_path=self.model_path))
        print("Model Dir: {model_dir}".format(model_dir=self.model_dir))
        print("Data Path: {data_path}".format(data_path=self.data_path))
        print("Data Dir: {data_dir}".format(data_dir=self.data_dir))
        if self.initial_dyanmic_data_paths is not None:
           for namedpath in self.initial_dynamic_data_dirs:
              print("Updatable data labled {name} is initially in {directory}".format(name= namedpath.name, directory=namedpath.path))

        print('loading model...')
        self.pre_and_post_processor = PreAndPostProcessor()
        self.oaas_wrapper = OaasWrapper("Model", is_llm_model=True)
        print('model loaded.')

    def Eval(self, data):
        data = json.loads(data)

        t0 = time.time()
        # Step 1: generate scene concepts
        (step1_prompts, metadata) = self.pre_and_post_processor.preprocess(data)
        step1_prompts = [step1_prompts] if isinstance(step1_prompts, str) else step1_prompts
        t1 = time.time()
        step1_outputs = self.oaas_wrapper.run(step1_prompts)
        t2 = time.time()

        # Step 2: expand each scene into a full prompt (batch of 5)
        (step2_prompts, metadata) = self.pre_and_post_processor.build_step2_prompts(
            step1_outputs, metadata
        )
        step2_prompts = [step2_prompts] if isinstance(step2_prompts, str) else step2_prompts
        t3 = time.time()
        step2_outputs = self.oaas_wrapper.run(step2_prompts)
        t4 = time.time()

        # Final: parse and structure results
        res = self.pre_and_post_processor.postprocess(step2_outputs, metadata)
        t5 = time.time()
        print(f"[TIMING] preprocess={t1-t0:.3f}s  step1_infer={t2-t1:.3f}s  build_step2={t3-t2:.3f}s  step2_infer={t4-t3:.3f}s  postprocess={t5-t4:.3f}s  total={t5-t0:.3f}s")
        return res

    def EvalBatch(self, data_list):
        all_step1_prompts = []
        all_metadata = []
        for i in range(len(data_list)):
            (p, m) = self.pre_and_post_processor.preprocess(json.loads(data_list[i]))
            all_step1_prompts.extend(p)
            all_metadata.append(m)

        # Step 1: batch all scene generation prompts
        step1_outputs = self.oaas_wrapper.run(all_step1_prompts)

        # Step 2: build expansion prompts from all Step 1 results
        all_step2_prompts = []
        all_step2_metadata = []
        for i in range(len(data_list)):
            step1_out = step1_outputs[i]
            (s2_prompts, s2_meta) = self.pre_and_post_processor.build_step2_prompts(
                step1_out, all_metadata[i]
            )
            all_step2_prompts.extend(s2_prompts)
            all_step2_metadata.append(s2_meta)

        # Step 2: batch all expansion prompts
        step2_outputs = self.oaas_wrapper.run(all_step2_prompts)

        # Postprocess: split step2_outputs back per request (5 scenes each)
        res = []
        offset = 0
        for i in range(len(data_list)):
            num_scenes = len(all_step2_metadata[i][0].get('scenes', []))
            request_outputs = step2_outputs[offset:offset + num_scenes]
            offset += num_scenes
            res.append(self.pre_and_post_processor.postprocess(
                request_outputs, all_step2_metadata[i]
            ))
        return json.dumps(res)

    def EvalBinary(self, data):
        return data

    def EvalBatchBinary(self, data_list):
        responses = []
        for i in range(0, len(data_list)):
            responses.append(data_list[i])
        return responses

    def OnDataUpdate(self, updated_paths):
        print('Got a fresh set of updated data')
        updated_dirs = utils.get_named_directories(updated_paths)
        for namedpath in updated_dirs:
              print("Updated data labled {name} is loaded in {directory}".format(name= namedpath.name, directory=namedpath.path))
