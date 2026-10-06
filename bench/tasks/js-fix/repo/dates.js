/**
 * Formats a duration given in whole seconds as a short human-readable string.
 *
 *   formatDuration(3723) // "1h 2m 3s"
 *   formatDuration(60)   // "1m"
 *   formatDuration(0)    // "0s"
 *
 * Units that are zero are omitted, except that a duration of 0 is "0s".
 */
function formatDuration(seconds) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor(seconds / 60);
  const secs = seconds % 60;

  const parts = [];
  if (hours > 0) parts.push(`${hours}h`);
  if (minutes > 0) parts.push(`${minutes}m`);
  if (secs > 0) parts.push(`${secs}s`);
  return parts.length > 0 ? parts.join(" ") : "0s";
}

module.exports = { formatDuration };
