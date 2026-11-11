"""Prepare private ENV_02 Test inputs before either experiment mode starts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from rsi.common import ROOT, write_json, record_path
from benchmarks.medrsee.membership import build_index, fingerprint
from benchmarks.medrsee.source import DATA
from benchmarks.medrsee.test_inputs import PreparedTestInputs, resolve_cache_root, sha


def check_inputs(data_root, cache_root, report_path, *, verify_only=False):
    """Prepare or verify host-private inputs without starting research or an API."""
    report_path = record_path(report_path)
    started = time.monotonic()
    try:
        index = build_index('ENV_02', ROOT / 'runs/medrsee-indices' / fingerprint() / 'ENV_02.sqlite')
        cache = PreparedTestInputs(resolve_cache_root(data_root, cache_root), 'ENV_02', index, data_root)
        def progress(value):
            if value['batches'] % 100 == 0 or value['completed'] == value['total']:
                value = dict(value, status='preparing', seconds=time.monotonic()-started,
                             labels_exported=False, exposed_to_agents=False)
                write_json(report_path, value)
                print(json.dumps(value), flush=True)
        result = dict(cache_hit=True, **cache.verify()) if verify_only else cache.prepare(progress)
        report = dict(status='ready', operation='verify' if verify_only else 'prepare',
                      checked_at=datetime.now(timezone.utc).isoformat(), task='ENV_02',
                      key=cache.key, cache_root=str(cache.root), entry=str(cache.entry),
                      samples=result['identity']['count'], batches=len(result['batches']), files=len(result['files']),
                      bytes=sum(f['bytes'] for f in result['files'].values()), cache_hit=result['cache_hit'],
                      manifest_sha256=sha(cache.entry/'manifest.json'),
                      seconds=time.monotonic()-started, labels_exported=False, exposed_to_agents=False)
        write_json(report_path, report)
        return report
    except Exception as exc:
        write_json(report_path, dict(status='failed', operation='verify' if verify_only else 'prepare',
                                    seconds=time.monotonic()-started, error=f'{type(exc).__name__}: {exc}',
                                    labels_exported=False, exposed_to_agents=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--cache-root')
    parser.add_argument('--data-root')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--report', required=True, type=record_path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else {}
    report = check_inputs(args.data_root or config.get('medrsee_data_root', str(DATA)),
                          args.cache_root or config.get('medrsee_test_cache_root'),
                          args.report, verify_only=args.verify_only)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
