#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate dataset of generated chains-of-resoning.

Example Usage:
python my_roscoe.py -p roscoe_inputs/sample.jsonl -t sim_sce -m facebook/roscoe-512-roberta-base
"""
import json
import os

from nltk.tokenize import sent_tokenize

from projects.roscoe.score import (
    SEQ_EMB_MODEL_TYPES,
    Chain,
    Evaluator,
    REASONING_SCORES,
    SENT_TRANS,
)

from parlai.core.params import ParlaiParser

DEFAULT_INPUT_PATH = f"./projects/roscoe/roscoe_data/generated/"
DEFAULT_OUTPUT_PATH = f"./projects/roscoe/scores/"

DATASETS = ["drop", "esnli", "cosmos", "gsm8k", "semeval"]


class ReasoningSteps(Chain):
    def __init__(self, line: str) -> None:
        self.chain = sent_tokenize(line.strip())


class ReasoningEvaluator(Evaluator):
    def __init__(
        self,
        model_type: str,
        transformer_model: str,
        discourse_batch: int,
        coherence_batch: int,
        **kwargs,
    ) -> None:
        super().__init__(
            hypos=[],
            context=[],
            references=[],
            model_type=model_type,
            transformer_model=transformer_model,
            discourse_batch=discourse_batch,
            coherence_batch=coherence_batch,
            **kwargs,
        )

    def update_evaluator(self, in_file: str):
        hypotheses = []
        contexts = []
        refs = []
        with open(in_file) as _f:
            for line in _f:
                jline = json.loads(line)
                hypothesis = ReasoningSteps(line=jline["hypothesis"])
                context = ReasoningSteps(line=jline["context"])
                reference = ReasoningSteps(line=jline["reference"])
                hypotheses.append(hypothesis)
                contexts.append(context)
                refs.append(reference)
        super().set_hypos(hypotheses)
        super().set_context(contexts)
        super().set_references(refs)


if __name__ == '__main__':
    parser = ParlaiParser(add_model_args=True)
    parser.add_argument(
        '--dataset-path',
        '-p',
        type=str,
        required=True,
        help='Path to files with predictions',
    )
    parser.add_argument(
        '--suffix',
        '-s',
        type=str,
        required=False,
        default="json",
        help='File suffix to match',
    )
    parser.add_argument(
        '--model-type',
        '-t',
        type=str,
        required=False,
        default=SENT_TRANS,
        choices=SEQ_EMB_MODEL_TYPES,
        help='Model type for embedding sequences.',
    )
    parser.add_argument(
        '--model-name',
        '-m',
        type=str,
        required=False,
        default="all-mpnet-base-v2",
        help='Transformer model name for embeddings. Must be compatible with model_type',
    )
    parser.add_argument(
        '--ppl-model-name',
        type=str,
        required=False,
        default="gpt2-large",
        help='Transformer HuggingFace model name for calculating perplexity-based metrics.',
    )
    parser.add_argument(
        '--discourse-batch',
        '-db',
        type=int,
        required=False,
        default=64,
        help='Batch size for discourse calculation',
    )
    parser.add_argument(
        '--coherence-batch',
        '-cb',
        type=int,
        required=False,
        default=16,
        help='Batch size for coherence calculation',
    )
    parser.add_argument(
        '--scores',
        type=str,
        nargs="*",
        default=REASONING_SCORES,
        choices=REASONING_SCORES,
        help=(
            'Scores to calculate. If the data is incompatible with a specified score '
            '(e.g. no reference is available) the score will be ignored.'
        ),
    )
    parser.add_argument(
        '--output-directory',
        type=str,
        default="roscoe_outputs",
        help='Where to save the scores.',
    )

    opt = parser.parse_args()
    evaluator = ReasoningEvaluator(
        score_types=opt['scores'],
        model_type=opt["model_type"],
        transformer_model=opt["model_name"],
        ppl_model=opt["ppl_model_name"],
        discourse_batch=opt["discourse_batch"],
        coherence_batch=opt["coherence_batch"],
    )

    input_file_path = opt['dataset_path']

    output_directory = os.path.join(opt['output_directory'], opt["model_name"].split('/')[-1])    
    os.makedirs(output_directory, exist_ok=True)
    out_p = os.path.join(
        output_directory, input_file_path.split('.')[0].replace('/', '__') + ".json"
    )

    print(f"Evaluating {input_file_path}")
    evaluator.update_evaluator(input_file_path)
    score_types = [st for st in REASONING_SCORES if st in opt['scores']]
    scores = evaluator.evaluate(score_types=score_types)

    safe_divide = lambda n, d: n / d if d > 0.0 else 0.0
    metrics = {
        metric_name: round(100 * safe_divide(sum(scores[metric_name]), len(scores[metric_name])), 1)
        if scores[metric_name] else 0.0 for metric_name in scores.keys()
    }
    print(f"Saving metrics in {out_p}")
    with open(out_p, "w") as file:
        json.dump(metrics, file, indent=4)
