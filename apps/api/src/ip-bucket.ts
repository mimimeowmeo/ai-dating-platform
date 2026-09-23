import { isIPv6 } from "node:net";

/** IPv6 用戶端通常握有整段前綴，逐位址限流換個位址就能繞過，所以以 /56 為一桶（同 express-rate-limit 的預設）。 */
export function ipBucket(ip: string | undefined): string {
  if (!ip) return "unknown";
  if (!isIPv6(ip)) return ip;
  const g = groups(ip);
  if (g.slice(0, 5).every((x) => x === 0) && g[5] === 0xffff)
    return [g[6] >> 8, g[6] & 255, g[7] >> 8, g[7] & 255].join(".");
  const hex = (x: number) => x.toString(16);
  return `${g.slice(0, 3).map(hex).join(":")}:${hex(g[3] & 0xff00)}::/56`;
}

function groups(ip: string): number[] {
  let text = ip.toLowerCase().split("%")[0];
  const v4 = text.match(/(\d+)\.(\d+)\.(\d+)\.(\d+)$/);
  if (v4) {
    const [a, b, c, d] = v4.slice(1).map(Number);
    text = `${text.slice(0, v4.index)}${((a << 8) | b).toString(16)}:${((c << 8) | d).toString(16)}`;
  }
  const [head, tail] = text.split("::");
  const left = head ? head.split(":") : [];
  const right = tail ? tail.split(":") : [];
  const fill =
    tail === undefined ? [] : Array(8 - left.length - right.length).fill("0");
  return [...left, ...fill, ...right].map((x) => parseInt(x, 16));
}
