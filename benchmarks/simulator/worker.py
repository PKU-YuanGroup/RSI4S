"""Trusted JSON process. No generated Agent code is imported here."""
import contextlib
import importlib.util
import json
import sys
from pathlib import Path


def clean(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def main(task, root):
    root = Path(root)
    path = root / ('trusted/episode.py' if task == 'MAT_01' else 'episode.py')
    spec = importlib.util.spec_from_file_location('simulator_episode', path)
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
        episode = module.Episode(root)
    terminal, metrics, revision = False, None, 0
    observation = (clean(episode.observe()) if task == 'CHEM_01' else
                   {'note': 'Use wh and get_sample_information for instrument observations.'})

    def status():
        if task == 'CHEM_01':
            usage = dict(steps=int(episode.env.unwrapped.steps), max_steps=50)
        else:
            usage = dict(commands=episode.used_commands, max_commands=128,
                         exposures=episode.used_exposures, max_exposures=64)
        return dict(task=task, observation=observation, terminal=terminal,
                    metrics=metrics, revision=revision, usage=usage)

    print(json.dumps({'result': status()}, allow_nan=False), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            op = request['op']
            with contextlib.redirect_stdout(sys.stderr):
                if op == 'action':
                    if terminal:
                        raise ValueError('Episode ended; reset is not permitted')
                    action, args = request['action'], request.get('arguments', {})
                    if not isinstance(args, dict):
                        raise ValueError('arguments must be an object')
                    # This also prevents boolean actions, reset, state inspection,
                    # and the organizer-only initial/final methods.
                    if task == 'CHEM_01':
                        if action != 'step' or set(args) != {'action'} or type(args['action']) is not int:
                            raise ValueError('Expected step with integer action 0..40')
                        observation = clean(episode.step(args['action']))
                        terminal = episode.done
                        if terminal:
                            metrics = episode.score()
                    elif action == 'finish':
                        if args:
                            raise ValueError('finish takes no arguments')
                        metrics = episode.finish()
                        terminal = True
                        observation = metrics
                    else:
                        observation = clean(episode.step(action, args))
                    revision += 1
                elif op != 'observe':
                    raise ValueError('Unsupported simulator operation')
            reply = {'result': status()}
        except Exception as exc:
            # Failed simulator operations may already have spent acquisition budget.
            revision += 1
            reply = {'error': f'{type(exc).__name__}: {exc}', 'result': status()}
        print(json.dumps(clean(reply), allow_nan=False), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
