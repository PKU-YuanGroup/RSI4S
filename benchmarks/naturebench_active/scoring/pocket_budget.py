"""Explicit, recorded revision of the native aggregate docking deadline only."""
import ast
import inspect
from pathlib import Path


def with_budget(function, seconds):
    if seconds is not None and (type(seconds) is not int or seconds < 3000):
        raise ValueError('Revised pocket docking budget must be null or an integer >= 3000 seconds')
    source=Path(inspect.getsourcefile(function))
    module=ast.parse(source.read_text())
    functions=[node for node in module.body if isinstance(node,ast.FunctionDef) and node.name==function.__name__]
    if len(functions)!=1:raise ValueError('Native evaluator function contract changed')
    # The released evaluator puts the deadline in calculate_metrics, called
    # by run_evaluation. Compile that helper in the same copied namespace so
    # the observed executor and compute_vina_score remain in effect.
    owners=[node for node in module.body if isinstance(node,ast.FunctionDef) and
            any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and
                t.id=='DOCKING_TIME_BUDGET' for t in n.targets) for n in ast.walk(node))]
    if len(owners)!=1 or owners[0].name not in (function.__name__,'calculate_metrics'):
        raise ValueError('Native docking deadline contract changed; inspect before revising it')
    if owners[0] is not functions[0] and not any(isinstance(n,ast.Call) and
            isinstance(n.func,ast.Name) and n.func.id==owners[0].name for n in ast.walk(functions[0])):
        raise ValueError('Native evaluation no longer calls the docking helper')
    body=owners if owners[0] is functions[0] else owners+functions
    tree=ast.Module(body=body,type_ignores=[])
    assignments=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and
                 any(isinstance(t,ast.Name) and t.id=='DOCKING_TIME_BUDGET' for t in n.targets)]
    if len(assignments)!=1 or ast.dump(assignments[0].value)!=ast.dump(ast.parse('50 * 60',mode='eval').body):
        raise ValueError('Native docking deadline contract changed; inspect before revising it')
    assignments[0].value=ast.Constant(seconds)
    if seconds is None:
        guards=[n for n in ast.walk(tree) if isinstance(n,ast.If) and
                ast.dump(n.test)==ast.dump(ast.parse('elapsed > DOCKING_TIME_BUDGET',mode='eval').body)]
        if len(guards)!=1:
            raise ValueError('Native docking cutoff guard changed; inspect before removing it')
        guards[0].test=ast.parse('DOCKING_TIME_BUDGET is not None and elapsed > DOCKING_TIME_BUDGET',mode='eval').body
    ast.fix_missing_locations(tree)
    scope=dict(function.__globals__)
    exec(compile(tree,inspect.getsourcefile(function),'exec'),scope)
    return scope[function.__name__]
