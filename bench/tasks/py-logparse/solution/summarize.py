import json
import re
from collections import Counter

LINE = re.compile(r'"\S+ (\S+) [^"]*" (\d{3}) ')

total, statuses, paths = 0, Counter(), Counter()
with open("access.log", encoding="utf-8") as f:
    for line in f:
        m = LINE.search(line)
        if m:
            total += 1
            paths[m.group(1)] += 1
            statuses[m.group(2)] += 1

summary = {
    "total": total,
    "by_status": dict(statuses),
    "top_paths": [[p, c] for p, c in sorted(paths.items(), key=lambda kv: (-kv[1], kv[0]))[:3]],
}
with open("summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
