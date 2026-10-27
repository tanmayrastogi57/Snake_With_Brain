import json
from datetime import datetime, timezone
from pathlib import Path


class JsonlLogger:
    """Append structured experiment records without keeping a file handle open."""

    def __init__(self, path, context=None):
        self.path = Path(path) if path else None
        self.context = dict(context or {})
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event, **fields):
        if self.path is None:
            return

        record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event': event,
            **self.context,
            **fields,
        }
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, sort_keys=True) + '\n')

