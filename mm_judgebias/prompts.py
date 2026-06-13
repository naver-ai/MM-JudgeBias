# MM-JudgeBias
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0

MAX_SCORE: int = 10

EVAL_PROMPT = f"""
### Task Description:
An instruction (might include an input inside it), a response to evaluate, and an image are given.

1. Write a detailed feedback that assesses the quality of the response strictly based on how well it follows the given instruction.
2. After writing a feedback, write a score that is an integer between 1 and {MAX_SCORE}.
3. Your response must adhere strictly to the following format: ### Feedback: (write a feedback) ### Score: (Only output a single integer between 1 and {MAX_SCORE}, without any additional text or explanation.)
4. Please do not generate any other opening, closing, and explanations.

### The instruction to evaluate:
{{instruction}}

### Response to evaluate:
{{response}}

### Feedback:
### Score:
"""
