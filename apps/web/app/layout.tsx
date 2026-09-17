import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "@/components/providers";

export const metadata: Metadata = {
  title: "遇見 meet — 每段故事，從真實開始",
  description: "從共同喜好開始，認識值得好好相處的人。遇見，讓真實的你被看見。",
};
export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-Hant">
      <body>
        <Providers>{children}</Providers>
        {process.env.FIGMA_CAPTURE === "1" && (
          <>
            <style>{"nextjs-portal { display: none !important; }"}</style>
            <script dangerouslySetInnerHTML={{ __html: `
              (() => {
                const timer = setInterval(() => {
                  const publicPage = ['/', '/login', '/register'].includes(location.pathname);
                  const ready = publicPage ? document.querySelector('h1') : document.querySelector('.app-shell h1');
                  if (!ready || document.querySelector('.spin')) return;
                  clearInterval(timer);
                  setTimeout(() => {
                    const s = document.createElement('script');
                    s.src = 'https://mcp.figma.com/mcp/html-to-design/capture.js';
                    s.async = true;
                    document.body.appendChild(s);
                  }, 2000);
                }, 250);
              })();
            ` }} />
          </>
        )}
      </body>
    </html>
  );
}
