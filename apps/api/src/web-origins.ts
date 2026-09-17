/** Only pair the two explicit loopback names; never trust arbitrary Host headers. */
export function webOrigins(configured: string): string[] {
  const url = new URL(configured);
  const origins = [url.origin];
  if (url.hostname === "localhost" || url.hostname === "127.0.0.1") {
    url.hostname = url.hostname === "localhost" ? "127.0.0.1" : "localhost";
    origins.push(url.origin);
  }
  return origins;
}
