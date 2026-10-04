"use strict";
// Hidden check for js-fix: formatDuration must follow its doc comment.
const assert = require("node:assert");
const path = require("node:path");

const { formatDuration } = require(path.join(__dirname, "..", "dates.js"));
assert.strictEqual(typeof formatDuration, "function", "dates.js must export formatDuration");

// The cases the doc comment spells out, plus the boundaries around each unit.
const cases = [
  [0, "0s"],
  [59, "59s"],
  [60, "1m"],
  [61, "1m 1s"],
  [3600, "1h"],
  [3723, "1h 2m 3s"],
  [86399, "23h 59m 59s"],
];
for (const [seconds, want] of cases) {
  assert.strictEqual(formatDuration(seconds), want, `formatDuration(${seconds})`);
}

// An independent reference for every second up to a day, so a partial fix can't slip through.
function reference(total) {
  const h = (total - (total % 3600)) / 3600;
  const m = ((total % 3600) - (total % 60)) / 60;
  const s = total % 60;
  const parts = [];
  if (h) parts.push(h + "h");
  if (m) parts.push(m + "m");
  if (s) parts.push(s + "s");
  return parts.join(" ") || "0s";
}
for (let t = 0; t < 86400; t++) {
  assert.strictEqual(formatDuration(t), reference(t), `formatDuration(${t})`);
}

console.log("ok");
