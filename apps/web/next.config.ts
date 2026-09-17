import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  poweredByHeader: false,
  async rewrites() {
    const backend = process.env.API_PROXY_URL || "http://127.0.0.1:3001";
    return [
      { source: "/api/:path*", destination: `${backend}/api/:path*` },
      {
        source: "/socket.io/:path*",
        destination: `${backend}/socket.io/:path*`,
      },
    ];
  },
};
export default nextConfig;
