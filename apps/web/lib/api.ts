import { create } from "zustand";
export type User = { id: string; email: string; isVerified: boolean };
export type Photo = {
  id: string;
  url: string;
  isAvatar: boolean;
  displayOrder: number;
};
export type Card = {
  userId: string;
  displayName: string;
  age: number;
  gender: string;
  bio: string;
  city: string;
  datingIntent: string;
  interests: string[];
  hobbies: string[];
  foods: string[];
  photos: Photo[];
  isVerified: boolean;
  // traits 資料表的 code：dating_goal 放 datingGoals，其餘類別放 traits。
  traits: string[];
  datingGoals: string[];
};
export type Profile = {
  userId: string;
  displayName: string;
  birthDate: string;
  gender: string;
  bio: string;
  city: string;
  latitude: number;
  longitude: number;
  datingIntent: string;
  heightCm?: number | null;
  occupation?: string | null;
  education?: string | null;
  interests: string[];
  hobbies: string[];
  foods: string[];
  photos: Photo[];
  traits: string[];
  datingGoals: string[];
};
export type Preferences = {
  minAge: number;
  maxAge: number;
  preferredGender: string;
  maxDistanceKm: number;
  minHeightCm: number;
  maxHeightCm: number;
  preferredDatingIntent: string;
};
export type Message = {
  id: string;
  conversationId: string;
  senderId: string;
  content: string;
  clientId: string;
  createdAt: string;
};
export type Conversation = {
  id: string;
  matchId: string;
  otherUser: Card;
  lastMessage: Message | null;
  unreadCount: number;
  otherLastReadAt: string | null;
};
export type Match = {
  id: string;
  createdAt: string;
  otherUser: Card;
  conversationId: string;
};
export const useAuth = create<{
  token: string | null;
  user: User | null;
  ready: boolean;
  revision: number;
  set: (token: string | null, user: User | null) => void;
}>((set) => ({
  token: null,
  user: null,
  ready: false,
  revision: 0,
  // 明確設定登入狀態（登入、登出、refresh 成功或被拒）就代表已知是否登入；暫時失敗不會呼叫這裡。
  set: (token, user) =>
    set((state) => ({
      token,
      user,
      ready: true,
      revision: state.revision + 1,
    })),
}));
let refreshPromise: Promise<boolean> | null = null;
export async function refresh() {
  if (!refreshPromise)
    refreshPromise = (async () => {
      const revision = useAuth.getState().revision;
      try {
        const r = await fetch("/api/v1/auth/refresh", {
          method: "POST",
          credentials: "include",
        });
        // 較早的 refresh 不得覆蓋使用者剛完成的登入或登出。
        if (useAuth.getState().revision !== revision)
          return !!useAuth.getState().token;
        // 只有 session 被拒（401／403）才代表已登出；限流（429）與伺服器錯誤只是暫時失敗，保留目前的登入狀態。
        if (r.status === 401 || r.status === 403) {
          useAuth.getState().set(null, null);
          return false;
        }
        if (!r.ok) return false;
        const d = await r.json();
        if (useAuth.getState().revision !== revision)
          return !!useAuth.getState().token;
        useAuth.getState().set(d.accessToken, d.user);
        return true;
      } catch {
        return false;
      } finally {
        refreshPromise = null;
      }
    })();
  return refreshPromise;
}
export async function api<T>(
  path: string,
  options: RequestInit = {},
  retry = true,
): Promise<T> {
  const headers = new Headers(options.headers);
  if (!(options.body instanceof FormData) && options.body)
    headers.set("Content-Type", "application/json");
  const token = useAuth.getState().token;
  if (token) headers.set("Authorization", `Bearer ${token}`);
  let response: Response;
  try {
    response = await fetch(`/api/v1${path}`, {
      ...options,
      headers,
      credentials: "include",
    });
  } catch {
    throw new Error("無法連線，請確認網路後再試。");
  }
  if (
    response.status === 401 &&
    retry &&
    !path.startsWith("/auth/") &&
    (await refresh())
  )
    return api(path, options, false);
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    // 超過 Nginx 上限時回傳的是 HTML 錯誤頁，沒有 JSON 訊息可顯示。
    throw new Error(
      data.message ||
        (response.status === 413
          ? "檔案太大，請選擇 8 MB 以內的圖片。"
          : "操作失敗，請稍後再試。"),
    );
  }
  const text = await response.text();
  // Nest may serialize a nullable result as an empty successful response.
  return JSON.parse(text || "null") as T;
}
export const send = <T>(path: string, body: unknown, method = "POST") =>
  api<T>(path, { method, body: JSON.stringify(body) });
export const intentLabels: Record<string, string> = {
  serious: "長期關係",
  casual: "輕鬆認識",
  friendship: "從朋友開始",
  any: "都可以",
};
export const genderLabels: Record<string, string> = {
  woman: "女性",
  man: "男性",
  nonbinary: "非二元性別",
  any: "不限性別",
};
