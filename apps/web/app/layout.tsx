import type { Metadata } from "next";
import { Noto_Sans_TC, Pacifico } from "next/font/google";
import "./globals.css";
import { Providers } from "@/components/providers";

const notoSansTC = Noto_Sans_TC({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-noto-tc",
});
const pacifico = Pacifico({
  weight: "400",
  subsets: ["latin"],
  display: "swap",
  variable: "--font-pacifico",
});

export const metadata: Metadata = {
  title: "HeartLink — 讓緣分，從這裡開始",
  description: "每一次相遇，都值得期待。從共同喜好開始，認識值得好好相處的人。",
};
export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="zh-Hant"
      className={`${notoSansTC.variable} ${pacifico.variable}`}
      suppressHydrationWarning
    >
      <head>
        {/* 在畫面出現前就決定日夜模式，避免先閃一下白底。 */}
        <script
          dangerouslySetInnerHTML={{
            __html: `(()=>{try{const t=localStorage.getItem("theme");document.documentElement.dataset.theme=t==="dark"||t==="light"?t:(matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light")}catch{document.documentElement.dataset.theme="light"}})()`,
          }}
        />
      </head>
      <body>
        <Providers>{children}</Providers>
        {process.env.FIGMA_CAPTURE === "1" && (
          <>
            <style>{"nextjs-portal { display: none !important; }"}</style>
            <script
              dangerouslySetInnerHTML={{
                __html: `
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
            `,
              }}
            />
          </>
        )}
      </body>
    </html>
  );
}
