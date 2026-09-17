"use client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { refresh, useAuth } from "@/lib/api";
export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: { retry: 1, staleTime: 15000, refetchOnWindowFocus: true },
        },
      }),
  );
  useEffect(() => {
    let cancelled = false;
    let retry: ReturnType<typeof setTimeout> | undefined;
    // 還無法確認是否登入（限流或伺服器暫時錯誤）時稍後重試，不把使用者當成已登出。
    const restore = (attempt: number) => {
      void refresh().then((ok) => {
        if (cancelled || ok || useAuth.getState().ready) return;
        retry = setTimeout(
          () => restore(attempt + 1),
          Math.min(30_000, 2_000 * 2 ** attempt),
        );
      });
    };
    restore(0);
    const timer = setInterval(
      () => {
        if (useAuth.getState().token) void refresh();
      },
      12 * 60 * 1000,
    );
    return () => {
      cancelled = true;
      clearTimeout(retry);
      clearInterval(timer);
    };
  }, []);
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
