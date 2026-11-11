"""Host-owned scientific validity instructions shared by every research stage."""

RESEARCH_RULES = """Scientific validity is mandatory. Invalid methods are strictly prohibited,
regardless of their score, speed, or remaining time. Do not implement, submit,
recommend, or propagate a method that violates the task requirements.

Use only task-permitted inputs and methods. Never extract, reconstruct, copy,
or recover hidden or unauthorized targets from input fields, auxiliary files, lookup tables,
other instances, reference outputs, external sources, or evaluator feedback.
Readable data are not automatically authorized for every use. Do not pass
forbidden information through successor code, prompts, history, or artifacts.
Do not modify or bypass evaluation, fabricate results, or exploit metric or
output-format loopholes instead of solving the defined scientific task.

Follow the task's input/output definition and data-use restrictions. Use the
provided training data to train, learn, fit, or optimize where required. Simple
classical models, legitimate non-parametric methods, and task-permitted
optimization or numerical algorithms are allowed; neural networks are not
required. If unexpected target information is found, do not use it and record
the issue. If a valid method cannot be completed, report the limitation instead
of substituting a forbidden shortcut. Preserve these rules in evolved agents.
The final selected submission is reviewed by the validity judge.
"""
