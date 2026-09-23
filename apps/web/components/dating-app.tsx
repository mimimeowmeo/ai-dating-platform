"use client";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  Fragment,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { io, Socket } from "socket.io-client";
import {
  ArrowRight,
  ArrowUpRight,
  Bell,
  Check,
  CheckCheck,
  ChevronLeft,
  Camera,
  Compass,
  Heart,
  Loader2,
  LogOut,
  MapPin,
  MessageCircle,
  Moon,
  RefreshCw,
  Send,
  Settings2,
  ShieldCheck,
  Sparkles,
  Sun,
  UserRound,
  X,
} from "lucide-react";
import {
  api,
  refresh,
  requestSuggestions,
  send,
  useAuth,
  genderLabels,
  type Card,
  type Profile,
  type Preferences,
  type Match,
  type SentLike,
  type Conversation,
  type Message,
  type ReplySuggestion,
  type User,
} from "@/lib/api";
import { HeartIcon, LogoMark } from "@/components/icons";
import {
  categoryTitle,
  datingGoalLimit,
  groupTraits,
  rememberTraitLabels,
  traitCategories,
  traitLabel,
  traitSnapshot,
  traitsOf,
  type TraitRow,
} from "@/lib/traits";
function useData<T>(path: string) {
  const user = useAuth((s) => s.user);
  return useQuery<T>({
    queryKey: [path, user?.id],
    queryFn: () => api<T>(path),
    enabled: !!user,
  });
}
// 選項清單來自後端 GET /traits；先用內建快照顯示，拿到資料後再更新。
function useTraitCatalog() {
  const user = useAuth((s) => s.user);
  const query = useQuery<TraitRow[]>({
    queryKey: ["/traits"],
    queryFn: () => api<TraitRow[]>("/traits"),
    enabled: !!user,
    staleTime: 60 * 60 * 1000,
    initialData: traitSnapshot,
    initialDataUpdatedAt: 0,
  });
  useEffect(() => {
    rememberTraitLabels(query.data);
  }, [query.data]);
  return query.data;
}
function ErrorText({ message }: { message?: string }) {
  return message ? (
    <p className="error" role="alert">
      {message}
    </p>
  ) : null;
}
function Loading() {
  return (
    <div className="empty">
      <Loader2 className="spin" />
      <p>正在準備你的心動…</p>
    </div>
  );
}
function Empty({
  title,
  text,
  link,
  label,
}: {
  title: string;
  text: string;
  link?: string;
  label?: string;
}) {
  return (
    <div className="empty">
      <div className="empty-icon">
        <Sparkles size={30} />
      </div>
      <h2>{title}</h2>
      <p>{text}</p>
      {link && (
        <Link className="button" href={link}>
          {label}
          <ArrowRight size={17} />
        </Link>
      )}
    </div>
  );
}
function Portrait({
  person,
  large = false,
  small = false,
}: {
  person: { displayName: string; photos?: { url: string }[] };
  large?: boolean;
  // small：聊天室訊息旁邊的小頭像（36px），沒有照片時退回顯示名字第一個字。
  small?: boolean;
}) {
  return (
    <div
      className={["portrait", large ? "large" : "", small ? "small" : ""]
        .filter(Boolean)
        .join(" ")}
    >
      {person.photos?.[0] ? (
        <img src={person.photos[0].url} alt={`${person.displayName}的照片`} />
      ) : (
        <span>{initialOf(person.displayName)}</span>
      )}
    </div>
  );
}
function Logo({ href, light = false }: { href: string; light?: boolean }) {
  return (
    <Link href={href} className={light ? "logo light" : "logo"}>
      <LogoMark />
      <span className="logo-word">HeartLink.</span>
    </Link>
  );
}
// 設計稿的文字頭像：中文名取最後一個字（小晴 → 晴），其他取第一個字母。
function initialOf(name: string) {
  const chars = Array.from(name.trim());
  const last = chars.at(-1) ?? "";
  return /\p{Script=Han}/u.test(last) ? last : (chars[0] ?? "").toUpperCase();
}
function Avatar({
  name,
  tone = 0,
  size,
}: {
  name: string;
  tone?: number;
  size?: "md" | "sm";
}) {
  return (
    <span
      className={["avatar", tone % 2 ? "violet" : "", size]
        .filter(Boolean)
        .join(" ")}
      aria-hidden="true"
    >
      {initialOf(name)}
    </span>
  );
}
const clock = (iso: string) =>
  new Date(iso).toLocaleTimeString("zh-TW", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
function daysAgo(iso: string) {
  const day = (d: Date) =>
    new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  return Math.round((day(new Date()) - day(new Date(iso))) / 86_400_000);
}
function listTime(iso?: string) {
  if (!iso) return "";
  const days = daysAgo(iso);
  if (days <= 0) return clock(iso);
  if (days === 1) return "昨天";
  return new Date(iso).toLocaleDateString(
    "zh-TW",
    days < 7 ? { weekday: "short" } : { month: "numeric", day: "numeric" },
  );
}
function dayLabel(iso: string) {
  const days = daysAgo(iso);
  const day =
    days <= 0
      ? "今天"
      : days === 1
        ? "昨天"
        : new Date(iso).toLocaleDateString("zh-TW", {
            month: "long",
            day: "numeric",
          });
  return `${day} ${clock(iso)}`;
}
// 按下喜歡的時間：一週內講幾天前，再久就寫日期。
function likedLabel(iso: string) {
  const days = daysAgo(iso);
  if (days <= 0) return "今天按下喜歡";
  if (days === 1) return "昨天按下喜歡";
  if (days < 7) return `${days} 天前按下喜歡`;
  return `${new Date(iso).toLocaleDateString("zh-TW", {
    month: "numeric",
    day: "numeric",
  })} 按下喜歡`;
}
function matchedLabel(iso: string) {
  const hours = (Date.now() - new Date(iso).getTime()) / 3_600_000;
  if (hours >= 24) return null;
  return hours < 1 ? "剛剛互相喜歡" : `${Math.floor(hours)} 小時前互相喜歡`;
}
function ageOf(birthDate: string) {
  const [y, m, d] = birthDate.slice(0, 10).split("-").map(Number);
  const now = new Date();
  const month = now.getMonth() + 1;
  const beforeBirthday = month < m || (month === m && now.getDate() < d);
  return now.getFullYear() - y - (beforeBirthday ? 1 : 0);
}
// 探索偏好拉桿的範圍，與後端 preferencesInput 的上下限一致。
const ageRange = [18, 130];
const heightRange = [130, 250];
const distanceRange = [1, 2000];
// 一條軌道兩個把手：用百分比標出目前選到的區間。
const rangeStyle = (
  [min, max]: number[],
  low: number,
  high: number,
): CSSProperties =>
  ({
    "--low": (low - min) / (max - min),
    "--high": (high - min) / (max - min),
  }) as CSSProperties;
// 一條軌道兩個把手。兩個原生 range 疊在一起時只有 22px 的把手接得到滑鼠，
// 軌道本身是死的，兩顆重疊時下面那顆也點不到。改由外框接管指標事件：
// 按在哪裡就把離得最近的那顆移過去，按住繼續拖也是同一顆，跟單顆拉桿一樣。
// 原生 input 留給鍵盤與螢幕閱讀器（Tab 進去用方向鍵微調），也維持 e2e 用 fill() 操作。
function RangePair({
  range,
  low,
  high,
  labels,
  onChange,
}: {
  range: number[];
  low: number;
  high: number;
  labels: [string, string];
  // 只回報「哪一顆、要到哪」，夾住另一顆交給上層用最新狀態處理，拖太快也不會反過來。
  onChange: (which: "low" | "high", value: number) => void;
}) {
  const [min, max] = range;
  const box = useRef<HTMLDivElement>(null);
  const lowInput = useRef<HTMLInputElement>(null);
  const highInput = useRef<HTMLInputElement>(null);
  // 正在拖的是哪一顆；兩顆重疊又剛好按在上面時先不決定，看第一下往哪邊拖。
  const active = useRef<"low" | "high" | "either" | null>(null);
  const valueAt = (clientX: number) => {
    const el = box.current;
    if (!el) return low;
    const rect = el.getBoundingClientRect();
    const thumb =
      parseFloat(getComputedStyle(el).getPropertyValue("--thumb")) || 22;
    // 把手中心的行程是扣掉把手寬度後的那一段（與 CSS 的軌道畫法一致）。
    const ratio =
      (clientX - rect.left - thumb / 2) / Math.max(1, rect.width - thumb);
    return Math.round(min + Math.min(1, Math.max(0, ratio)) * (max - min));
  };
  const move = (which: "low" | "high", value: number) => {
    onChange(which, value);
    (which === "low" ? lowInput : highInput).current?.focus({
      preventScroll: true,
    });
  };
  const stop = (e: React.PointerEvent<HTMLDivElement>) => {
    active.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId))
      e.currentTarget.releasePointerCapture(e.pointerId);
  };
  return (
    <div
      ref={box}
      className="range-pair"
      style={rangeStyle(range, low, high)}
      // 外框在 <label> 裡；擋掉 label 的預設行為，點軌道時焦點才不會被搶去第一個 input。
      onClick={(e) => e.preventDefault()}
      onPointerDown={(e) => {
        if (e.button !== 0) return;
        e.preventDefault();
        const value = valueAt(e.clientX);
        active.current =
          value < low
            ? "low"
            : value > high
              ? "high"
              : low === high
                ? "either"
                : value - low <= high - value
                  ? "low"
                  : "high";
        e.currentTarget.setPointerCapture(e.pointerId);
        if (active.current !== "either") move(active.current, value);
      }}
      onPointerMove={(e) => {
        if (!active.current) return;
        const value = valueAt(e.clientX);
        if (active.current === "either") {
          if (value === low) return;
          active.current = value > low ? "high" : "low";
        }
        move(active.current, value);
      }}
      onPointerUp={stop}
      onPointerCancel={stop}
    >
      <input
        ref={lowInput}
        type="range"
        aria-label={labels[0]}
        min={min}
        max={max}
        value={low}
        onChange={(e) => onChange("low", Number(e.target.value))}
      />
      <input
        ref={highInput}
        type="range"
        aria-label={labels[1]}
        min={min}
        max={max}
        value={high}
        onChange={(e) => onChange("high", Number(e.target.value))}
      />
    </div>
  );
}
const goalText = (codes?: string[]) =>
  codes?.length ? codes.map(traitLabel).join("、") : "";
// 新帳號要先補齊這幾項才能進站；照片必須先存好基本資料才能上傳。
const onboardingSteps = (profile?: Profile | null) => [
  {
    label: "基本資料",
    done: !!profile?.displayName && !!profile.city && !!profile.bio.trim(),
  },
  { label: "想遇見的關係", done: !!profile?.datingGoals.length },
  { label: "我的小熱愛", done: !!profile?.traits.length },
  { label: "一張生活照", done: !!profile?.photos.length },
];
const onboardingLeft = (profile?: Profile | null) =>
  onboardingSteps(profile).filter((step) => !step.done);
const nav = [
  { href: "/discover", label: "探索心動", icon: Compass },
  { href: "/matches", label: "我的配對", icon: Heart },
  { href: "/messages", label: "聊天室", icon: MessageCircle },
  { href: "/profile", label: "個人檔案", icon: UserRound },
];
// 側欄下半部：設定類的入口，與舊版位置一致。
const sideExtras = [
  { href: "/preferences", label: "探索偏好", icon: Settings2 },
  { href: "/verification", label: "真人驗證", icon: ShieldCheck },
];
type NavEntry = {
  href: string;
  label: string;
  icon: (props: { size?: number }) => React.ReactNode;
};
// 側欄項目：沿用舊版的 icon + 文字，目前所在的頁面右側加一個小圓點。
function navItem(n: NavEntry, locked: boolean, path: string, size = 22) {
  const active = path.startsWith(n.href);
  if (locked && n.href !== "/profile")
    return (
      <span
        key={n.href}
        className="nav-item locked"
        aria-disabled="true"
        title="完成個人檔案後解鎖"
      >
        <n.icon size={size} />
        <span>{n.label}</span>
      </span>
    );
  return (
    <Link
      key={n.href}
      href={n.href}
      title={n.label}
      aria-current={active ? "page" : undefined}
      className={active ? "nav-item active" : "nav-item"}
    >
      <n.icon size={size} />
      <span>{n.label}</span>
      {active && <i />}
    </Link>
  );
}
export function DatingApp() {
  const path = usePathname();
  const router = useRouter();
  const { user, ready } = useAuth();
  const client = useQueryClient();
  const [socket, setSocket] = useState<Socket | null>(null);
  const token = useAuth((s) => s.token);
  const profile = useData<Profile | null>("/profile");
  // 個人檔案沒補齊之前只能待在個人檔案頁，其他頁面與連結都先鎖住。
  const locked = profile.isSuccess && onboardingLeft(profile.data).length > 0;
  useEffect(() => {
    if (locked && !["/", "/login", "/register", "/profile"].includes(path))
      router.replace("/profile");
  }, [locked, path, router]);
  useEffect(() => {
    if (!ready) return;
    if (!user && !["/", "/login", "/register"].includes(path))
      router.replace("/login");
    if (user && path === "/login") router.replace("/discover");
    if (user && path === "/register") router.replace("/profile");
  }, [path, user, ready, router]);
  useEffect(() => {
    if (!token) return;
    const connection = io({
      path: "/socket.io",
      auth: { token },
      transports: ["websocket", "polling"],
      // 沒開這個選項時，WebSocket 被防火牆或代理擋下不會改用 polling，只會一直重試 WebSocket。
      tryAllTransports: true,
    });
    setSocket(connection);
    const invalidate = () => {
      void client.invalidateQueries({
        predicate: (q) =>
          ["/conversations", "/matches", "/likes", "/notifications"].includes(
            q.queryKey[0] as string,
          ),
      });
    };
    connection.on("notification:new", invalidate);
    connection.on("message:new", invalidate);
    connection.on("conversation:closed", () => {
      void client.invalidateQueries();
    });
    let attempts = 0;
    let retry: ReturnType<typeof setTimeout> | undefined;
    const recover = () => {
      clearTimeout(retry);
      retry = setTimeout(
        () => {
          void refresh().then((ok) => {
            if (
              ok &&
              useAuth.getState().token === token &&
              !connection.connected
            )
              connection.connect();
          });
        },
        Math.min(30_000, 1_000 * 2 ** attempts++),
      );
    };
    connection.on("connect", () => {
      attempts = 0;
    });
    connection.on("disconnect", (reason) => {
      if (reason === "io server disconnect") recover();
    });
    connection.on("connect_error", () => {
      if (!connection.active) recover();
    });
    return () => {
      clearTimeout(retry);
      connection.disconnect();
      setSocket(null);
    };
  }, [token, client]);
  if (path === "/") return <Landing />;
  if (path === "/login" || path === "/register")
    return <AuthPage register={path === "/register"} />;
  if (!ready || !user) return <Loading />;
  const logout = async () => {
    try {
      await send("/auth/logout", {});
    } finally {
      useAuth.getState().set(null, null);
      client.clear();
      router.push("/login");
    }
  };
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Logo href={locked ? "/profile" : "/discover"} />
        <p className="sidebar-tagline">每一次相遇，都值得期待</p>
        <span className="eyebrow nav-label">YOUR NEXT CHAPTER</span>
        <nav className="side-nav" aria-label="主要頁面">
          {nav.map((n) => navItem(n, locked, path))}
        </nav>
        <div className="sidebar-bottom">
          {sideExtras.map((n) => navItem(n, locked, path, 20))}
          <button type="button" className="nav-item" onClick={logout}>
            <LogOut size={20} />
            <span>登出</span>
          </button>
          <SidebarUser user={user} />
        </div>
      </aside>
      <div className="workspace">
        <header className="topbar">
          <Logo href={locked ? "/profile" : "/discover"} />
          <p className="topbar-tagline">讓緣分，從這裡開始。</p>
          <TopActions user={user} onLogout={logout} locked={locked} />
        </header>
        <main className="main-content">
          {path === "/discover" ? (
            <Discover />
          ) : path === "/profile" ? (
            <ProfilePage />
          ) : path === "/preferences" ? (
            <PreferencesPage />
          ) : path === "/verification" ? (
            <VerificationPage />
          ) : path === "/matches" ? (
            <MatchesPage />
          ) : path === "/notifications" ? (
            <NotificationsPage />
          ) : path.startsWith("/messages") ? (
            <MessagesPage id={path.split("/")[2]} socket={socket} />
          ) : (
            <Empty
              title="這個頁面還沒出現"
              text="回到探索，繼續你的故事。"
              link="/discover"
              label="回到探索"
            />
          )}
        </main>
      </div>
      <nav className="mobile-nav" aria-label="主要頁面">
        {nav.map((n) => {
          const active = path.startsWith(n.href);
          return locked && n.href !== "/profile" ? (
            <span key={n.href} className="locked" aria-disabled="true">
              <n.icon size={21} />
              {n.label}
            </span>
          ) : (
            <Link
              key={n.href}
              href={n.href}
              aria-current={active ? "page" : undefined}
              className={active ? "active" : ""}
            >
              <n.icon size={21} />
              {n.label}
            </Link>
          );
        })}
      </nav>
    </div>
  );
}
function SidebarUser({ user }: { user: User }) {
  const profile = useData<Profile | null>("/profile");
  return (
    <p className="side-user">
      <span>{profile.data?.displayName || user.email.split("@")[0]}</span>
      <span className="side-user-status">
        {" · "}
        <Link href="/verification">
          {user.isVerified ? "已驗證" : "尚未驗證"}
        </Link>
      </span>
    </p>
  );
}
// 日夜模式：實際的切換寫在 <html data-theme>，重新整理前由 layout 的小腳本先套用。
function ThemeToggle() {
  const [dark, setDark] = useState(false);
  useEffect(() => {
    setDark(document.documentElement.dataset.theme === "dark");
  }, []);
  const toggle = () => {
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("theme", next);
    } catch {
      // 無痕視窗不能寫入時，至少這一次切換仍然生效。
    }
    setDark(!dark);
  };
  return (
    <button
      type="button"
      className="icon-button"
      onClick={toggle}
      aria-label={dark ? "切換到日間模式" : "切換到夜間模式"}
    >
      {dark ? <Sun size={19} /> : <Moon size={19} />}
    </button>
  );
}
function TopActions({
  user,
  onLogout,
  locked,
}: {
  user: User;
  onLogout: () => void;
  locked: boolean;
}) {
  const profile = useData<Profile | null>("/profile");
  const notifications = useData<{ readAt: string | null }[]>("/notifications");
  const unread = notifications.data?.filter((n) => !n.readAt).length || 0;
  return (
    <div className="top-actions">
      <ThemeToggle />
      {locked ? (
        <span className="icon-button locked" aria-disabled="true" title="通知">
          <Bell size={19} />
        </span>
      ) : (
        <Link href="/notifications" className="icon-button" aria-label="通知">
          <Bell size={19} />
          {unread > 0 && <span className="count">{unread}</span>}
        </Link>
      )}
      <button
        type="button"
        className="icon-button"
        aria-label="登出"
        onClick={onLogout}
      >
        <LogOut size={19} />
      </button>
      <Link href="/profile" className="top-avatar" aria-label="我的個人檔案">
        {profile.data ? (
          <Portrait person={profile.data} />
        ) : (
          <Avatar name={user.email} size="sm" />
        )}
      </Link>
    </div>
  );
}
function Landing() {
  const user = useAuth((s) => s.user);
  return (
    <div className="landing">
      <header>
        <Logo href="/" />
        <Link className="button secondary" href={user ? "/discover" : "/login"}>
          {user ? "開始探索" : "登入"}
          <ArrowUpRight size={18} />
        </Link>
      </header>
      <main className="hero">
        <div className="hero-copy">
          <span className="eyebrow">
            <i className="dot" /> MAKE ROOM FOR SOMETHING REAL
          </span>
          <h1>
            不是找一個人。
            <br />
            是遇見<span>懂你的人。</span>
          </h1>
          <p>
            從一杯咖啡、一段旅行、一個共同喜好開始。
            <br />
            讓真實的你，遇見剛剛好的關係。
          </p>
          <Link className="button big" href={user ? "/discover" : "/register"}>
            開始你的故事
            <ArrowRight size={20} />
          </Link>
          <div className="hero-detail">
            <ShieldCheck size={18} />
            <span>彼此喜歡，才開始對話</span>
            <span>・</span>
            <span>你的步調，由你決定</span>
          </div>
        </div>
        <div className="hero-art" aria-hidden="true">
          <div className="orbit orbit-one" />
          <div className="orbit orbit-two" />
          <div className="abstract-card card-one">
            <span>
              little things,
              <br />
              great connections.
            </span>
            <div className="flower">✳</div>
            <small>共同喜好，是故事的起點。</small>
          </div>
          <div className="abstract-card card-two">
            <HeartIcon size={58} />
            <span>
              HELLO,
              <br />
              SOMETHING
              <br />
              <em>real.</em>
            </span>
          </div>
          <div className="floating-note">
            <Sparkles size={18} /> 每一次相遇，都值得期待。
          </div>
        </div>
      </main>
      <section className="values">
        <div>
          <span>01 / 真實表達</span>
          <h3>先讓自己被看見</h3>
          <p>分享你的生活、興趣與期待。</p>
        </div>
        <div>
          <span>02 / 雙向選擇</span>
          <h3>剛好，你也喜歡我</h3>
          <p>互相喜歡後，才開啟聊天。</p>
        </div>
        <div>
          <span>03 / 自在相處</span>
          <h3>每段關係都有自己的速度</h3>
          <p>自由調整偏好，也能隨時結束對話。</p>
        </div>
      </section>
      <footer>
        <span className="logo-word">HeartLink.</span>
        <span>讓緣分，從這裡開始。</span>
      </footer>
    </div>
  );
}
const authSchema = z.object({
  email: z.string().email("請填寫有效的電子郵件"),
  password: z
    .string()
    .min(12, "密碼至少 12 個字元")
    .max(72, "密碼最多 72 個字元"),
});
function AuthPage({ register }: { register: boolean }) {
  const router = useRouter();
  const [error, setError] = useState("");
  const form = useForm<z.infer<typeof authSchema>>({
    resolver: zodResolver(authSchema),
  });
  const submit = form.handleSubmit(async (values) => {
    setError("");
    try {
      const d = await send<{ accessToken: string; user: User }>(
        register ? "/auth/register" : "/auth/login",
        values,
      );
      useAuth.getState().set(d.accessToken, d.user);
      router.push(register ? "/profile" : "/discover");
    } catch (e) {
      setError((e as Error).message);
    }
  });
  return (
    <div className="auth-layout">
      <aside>
        <Logo href="/" light />
        <div>
          <span className="eyebrow">A LITTLE HELLO GOES A LONG WAY</span>
          <h1>
            有些美好，
            <br />
            從一句你好開始。
          </h1>
          <div className="auth-flower">
            <HeartIcon size={120} />
          </div>
        </div>
        <small>每一次相遇，都值得期待。</small>
      </aside>
      <main>
        <Link className="text-link" href="/">
          <ChevronLeft size={16} />
          回到首頁
        </Link>
        <div className="auth-form">
          <span className="eyebrow">
            {register ? "YOUR STORY STARTS HERE" : "GOOD TO SEE YOU AGAIN"}
          </span>
          <h1>{register ? "很高興，即將遇見你。" : "歡迎回來。"}</h1>
          <p>
            {register
              ? "建立帳號，慢慢認識值得相處的人。"
              : "你的下一段故事，正在等你。"}
          </p>
          <form onSubmit={submit}>
            <label>
              電子郵件
              <input
                type="email"
                autoComplete="email"
                placeholder="you@example.com"
                {...form.register("email")}
              />
            </label>
            <ErrorText message={form.formState.errors.email?.message} />
            <label>
              密碼
              <input
                type="password"
                autoComplete={register ? "new-password" : "current-password"}
                placeholder="至少 12 個字元"
                {...form.register("password")}
              />
            </label>
            <ErrorText message={form.formState.errors.password?.message} />
            {register && (
              <label className="check-label">
                <input type="checkbox" required />
                我已年滿 18 歲，了解這是本機開發版本。
              </label>
            )}
            <ErrorText message={error} />
            <button
              className="button full"
              disabled={form.formState.isSubmitting}
            >
              {form.formState.isSubmitting ? (
                <Loader2 className="spin" size={18} />
              ) : register ? (
                "建立帳號"
              ) : (
                "登入"
              )}
              <ArrowRight size={18} />
            </button>
          </form>
          <p className="auth-switch">
            {register ? "已經有帳號？" : "第一次來到 HeartLink？"}
            <Link href={register ? "/login" : "/register"}>
              {register ? "立即登入" : "建立帳號"}
            </Link>
          </p>
        </div>
      </main>
    </div>
  );
}
function Heading({
  overline,
  title,
  text,
  heart = false,
  children,
}: {
  overline: string;
  title: string;
  text: string;
  heart?: boolean;
  children?: React.ReactNode;
}) {
  return (
    <div className={children ? "page-heading has-actions" : "page-heading"}>
      <span className="eyebrow">{overline}</span>
      <h1>
        {title}
        {heart && <HeartIcon className="title-heart" />}
      </h1>
      <p>{text}</p>
      {children}
    </div>
  );
}
type Person = Pick<
  Card,
  | "displayName"
  | "age"
  | "city"
  | "bio"
  | "datingIntent"
  | "traits"
  | "datingGoals"
  | "photos"
  | "isVerified"
>;
function PersonCard({
  person,
  children,
}: {
  person: Person;
  children?: React.ReactNode;
}) {
  const tags = person.traits ?? [];
  return (
    <article className="person-card">
      <div className="person-photo">
        <Portrait person={person} large />
      </div>
      <div className="person-info">
        <span className="eyebrow">NICE TO MEET YOU</span>
        <h2>
          {person.displayName}
          <span>{person.age}</span>
          {person.isVerified && (
            <ShieldCheck size={24} aria-label="已通過驗證" />
          )}
        </h2>
        <p className="person-meta">
          <MapPin size={15} />
          {[person.city, goalText(person.datingGoals)]
            .filter(Boolean)
            .join(" · ")}
        </p>
        <p className="field-label">關於我</p>
        <p className="person-bio">
          {person.bio || "有些故事，適合在對話裡慢慢認識。"}
        </p>
        {tags.length > 0 && (
          <>
            <p className="field-label">我的小小熱愛</p>
            <div className="tags hash">
              {tags.map((code) => (
                <span key={code}>{traitLabel(code)}</span>
              ))}
            </div>
          </>
        )}
        {children}
      </div>
    </article>
  );
}
function PersonDialog({
  person,
  onClose,
}: {
  person: Person;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const current = dialog.current;
    if (current && !current.open) current.showModal();
    return () => current?.close();
  }, []);
  return (
    <dialog
      ref={dialog}
      className="person-dialog"
      aria-label={`${person.displayName}的檔案`}
      onClose={(e) => {
        // 開發模式的 StrictMode 會先關再開；只有真的關閉（例如按 Esc）才通知上層。
        if (!e.currentTarget.open) onClose();
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="dialog-body">
        <button
          type="button"
          className="dialog-close"
          aria-label="關閉"
          onClick={onClose}
        >
          <X size={18} />
        </button>
        <PersonCard person={person} />
      </div>
    </dialog>
  );
}
function Discover() {
  const query = useData<Card[]>("/discovery");
  const profile = useData<Profile | null>("/profile");
  const client = useQueryClient();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [match, setMatch] = useState(false);
  const person = query.data?.[0];
  async function act(action: "like" | "pass") {
    if (!person) return;
    setBusy(true);
    setError("");
    try {
      const result = await send<{ matched: boolean }>("/interactions", {
        targetUserId: person.userId,
        action,
      });
      setMatch(result.matched);
      await client.invalidateQueries({ queryKey: ["/discovery"] });
      await client.invalidateQueries({ queryKey: ["/matches"] });
      await client.invalidateQueries({ queryKey: ["/likes"] });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <Heading
        overline="A LITTLE SPARK"
        title="探索心動"
        heart
        text="在日常裡，遇見一點不一樣的心動。"
      >
        <Link className="button secondary" href="/preferences">
          探索偏好
        </Link>
      </Heading>
      <div className="discover-layout">
        <section className="discover-main" aria-label="為你探索">
          <ErrorText message={error || query.error?.message} />
          {match && (
            <div className="success" role="status">
              <HeartIcon filled size={20} />
              你們互相喜歡！<Link href="/matches">開始聊聊 →</Link>
              <button aria-label="關閉配對提示" onClick={() => setMatch(false)}>
                <X size={16} />
              </button>
            </div>
          )}
          {query.isLoading ? (
            <Loading />
          ) : !profile.data ? (
            <Empty
              title="先讓大家認識你"
              text="完成個人資料與城市，開始探索彼此適合的人。"
              link="/profile"
              label="完成個人檔案"
            />
          ) : !person ? (
            <Empty
              title="把美好的相遇，留給下一次"
              text="目前沒有符合雙方偏好的人。試著調整探索範圍，或邀請朋友建立帳號。"
              link="/preferences"
              label="調整探索範圍"
            />
          ) : (
            <PersonCard person={person}>
              <div className="discovery-actions">
                <button
                  className="action-button pass"
                  disabled={busy}
                  onClick={() => act("pass")}
                >
                  <X size={26} strokeWidth={1.5} />
                  略過
                </button>
                <button
                  className="action-button like"
                  disabled={busy}
                  onClick={() => act("like")}
                >
                  <HeartIcon filled size={26} />
                  喜歡
                </button>
              </div>
            </PersonCard>
          )}
          <p className="soft-note">
            <HeartIcon size={15} />
            慢慢認識，不急著心動。
          </p>
        </section>
        <DiscoverAside />
      </div>
    </>
  );
}
function DiscoverAside() {
  const matches = useData<Match[]>("/matches");
  const conversations = useData<Conversation[]>("/conversations");
  return (
    <aside className="discover-aside">
      <section className="side-card">
        <h2>
          新的連結
          <HeartIcon size={24} />
        </h2>
        <p className="side-card-note">有人也想多認識你一點</p>
        {matches.data?.length ? (
          <ul className="mini-matches">
            {matches.data.slice(0, 3).map((m) => (
              <li key={m.id}>
                <Link href={`/messages/${m.conversationId}`}>
                  <Portrait person={m.otherUser} />
                  <span>{m.otherUser.displayName}</span>
                </Link>
              </li>
            ))}
          </ul>
        ) : (
          <p className="side-card-note">互相喜歡後，會在這裡相遇。</p>
        )}
        <Link className="button secondary" href="/matches">
          查看所有配對
        </Link>
      </section>
      <section className="side-card">
        <h2>最近的對話</h2>
        {conversations.data?.length ? (
          <ul className="mini-conversations">
            {conversations.data.slice(0, 3).map((c) => (
              <li key={c.id}>
                <Link href={`/messages/${c.id}`}>
                  <b>{c.otherUser.displayName}</b>
                  <span>{c.lastMessage?.content || "從一句你好開始吧。"}</span>
                </Link>
              </li>
            ))}
          </ul>
        ) : (
          <p className="side-card-note">配對成功後，就能開始聊天。</p>
        )}
        <Link className="button secondary" href="/messages">
          前往聊天室
        </Link>
      </section>
    </aside>
  );
}
const cities = [
  { name: "台北市", lat: 25.033, lng: 121.5654 },
  { name: "新北市", lat: 25.012, lng: 121.4657 },
  { name: "桃園市", lat: 24.9937, lng: 121.301 },
  { name: "台中市", lat: 24.1477, lng: 120.6736 },
  { name: "台南市", lat: 22.9997, lng: 120.227 },
  { name: "高雄市", lat: 22.6273, lng: 120.3014 },
  { name: "新竹市", lat: 24.8138, lng: 120.9675 },
  { name: "花蓮市", lat: 23.991, lng: 121.6112 },
];
const photoLimit = 8 * 1024 * 1024;
const photoTooLarge = "照片太大，請選擇 8 MB 以內的檔案。";
function OnboardingPanel({
  profile,
  notice,
}: {
  profile?: Profile | null;
  notice: string;
}) {
  return (
    <section className="panel onboarding">
      <h2>完成這幾項，就能開始探索</h2>
      <p className="muted">
        第一次使用要先把個人檔案填完，其他頁面會在完成後解鎖。
      </p>
      {notice && (
        <p className="success" role="status">
          <Check size={18} />
          {notice}
        </p>
      )}
      <ul className="onboarding-steps">
        {onboardingSteps(profile).map((step) => (
          <li key={step.label} className={step.done ? "done" : ""}>
            {step.done ? (
              <Check size={16} />
            ) : (
              <span className="step-dot" aria-hidden="true" />
            )}
            {step.label}
          </li>
        ))}
      </ul>
    </section>
  );
}
function ProfilePage() {
  const query = useData<Profile | null>("/profile");
  const onboarding = query.isSuccess && onboardingLeft(query.data).length > 0;
  const [editing, setEditing] = useState(false);
  const [notice, setNotice] = useState("");
  // 手機版的編輯按鈕在頁面底部，切換檢視／編輯時回到頂端。
  const show = (edit: boolean, message = "") => {
    setEditing(edit);
    setNotice(message);
    window.scrollTo({ top: 0 });
  };
  // 最後一項補齊的當下留在表單，才不會把還沒儲存的輸入丟掉。
  // 用 effect 會先 render 出檢視模式、把表單卸載，所以在 render 當下就調整狀態。
  const [wasOnboarding, setWasOnboarding] = useState(onboarding);
  if (wasOnboarding !== onboarding) {
    setWasOnboarding(onboarding);
    if (!onboarding) {
      setEditing(true);
      setNotice("個人檔案完成了，現在可以開始探索。");
    }
  }
  return (
    <>
      <Heading
        overline="THIS IS ME"
        title="個人檔案"
        text="讓對的人，看見最真實的你。"
      />
      {onboarding ? (
        <OnboardingPanel profile={query.data} notice={notice} />
      ) : editing && notice ? (
        <p className="panel success" role="status">
          <Check size={18} />
          {notice}
        </p>
      ) : null}
      {query.isLoading ? (
        <Loading />
      ) : query.data && !editing && !onboarding ? (
        <ProfileView
          profile={query.data}
          notice={notice}
          onEdit={() => show(true)}
        />
      ) : (
        <ProfileForm
          profile={query.data ?? null}
          loadError={query.error?.message}
          onboarding={onboarding}
          onSaved={(message) => show(false, message)}
          onCancel={query.data && !onboarding ? () => show(false) : undefined}
        />
      )}
    </>
  );
}
function ProfileView({
  profile: p,
  notice,
  onEdit,
}: {
  profile: Profile;
  notice: string;
  onEdit: () => void;
}) {
  const user = useAuth((s) => s.user)!;
  const [preview, setPreview] = useState(false);
  const person: Person = {
    ...p,
    age: ageOf(p.birthDate),
    isVerified: user.isVerified,
  };
  const tags = p.traits ?? [];
  const catalog = useTraitCatalog();
  const lifePhotos = p.photos.slice(1);
  const pills = (codes: string[]) => (
    <div className="tags pills">
      {codes.map((code) => (
        <span key={code}>{traitLabel(code)}</span>
      ))}
    </div>
  );
  const actions = (
    <div className="profile-actions">
      <button type="button" className="button" onClick={onEdit}>
        編輯個人檔案
      </button>
      <button
        type="button"
        className="button secondary preview-button"
        onClick={() => setPreview(true)}
      >
        預覽公開頁面
      </button>
      <Link className="button secondary" href="/verification">
        {user.isVerified ? "已通過真人驗證" : "前往真人驗證"}
      </Link>
    </div>
  );
  return (
    <section className="profile-card">
      {notice && (
        <p className="success" role="status">
          <Check size={18} />
          {notice}
        </p>
      )}
      <div className="profile-top">
        <div className="profile-photo">
          <Portrait person={p} large />
        </div>
        <div className="profile-summary">
          <span className="eyebrow">MY PROFILE</span>
          <h2>
            {p.displayName}，{person.age}
          </h2>
          <p className="profile-city">{p.city}</p>
          <p className="profile-tagline">
            {[goalText(p.datingGoals), p.occupation]
              .filter(Boolean)
              .join(" · ")}
          </p>
          {tags.length > 0 && <div className="summary-tags">{pills(tags)}</div>}
          {actions}
        </div>
      </div>
      <div className="profile-details">
        <div className="profile-about">
          <h3>關於我</h3>
          <p>{p.bio || "還沒有寫下自我介紹。"}</p>
        </div>
        <div className="profile-loves">
          <h3>我的小熱愛</h3>
          {tags.length ? (
            groupTraits(catalog, tags).map((group) => (
              <div className="trait-group" key={group.category}>
                {group.category && <h4>{categoryTitle(group.category)}</h4>}
                {pills(group.codes)}
              </div>
            ))
          ) : (
            <p className="muted">還沒有選擇喜好。</p>
          )}
        </div>
        <div className="profile-life">
          <h3>我的生活照</h3>
          {lifePhotos.length ? (
            <div className="life-photos">
              {lifePhotos.map((photo) => (
                <img key={photo.id} src={photo.url} alt="我的生活照" />
              ))}
            </div>
          ) : (
            <p className="muted">加入更多照片，讓大家更認識你。</p>
          )}
        </div>
      </div>
      <div className="profile-actions-end">{actions}</div>
      {preview && (
        <PersonDialog person={person} onClose={() => setPreview(false)} />
      )}
    </section>
  );
}
function ProfileForm({
  profile,
  loadError,
  onboarding,
  onSaved,
  onCancel,
}: {
  profile: Profile | null;
  loadError?: string;
  onboarding: boolean;
  onSaved: (message: string) => void;
  onCancel?: () => void;
}) {
  const client = useQueryClient();
  const form = useForm({
    defaultValues: {
      displayName: profile?.displayName ?? "",
      birthDate: profile?.birthDate ?? "",
      gender: profile?.gender ?? "woman",
      bio: profile?.bio ?? "",
      city: profile?.city ?? "台北市",
      heightCm: profile?.heightCm ? String(profile.heightCm) : "",
      occupation: profile?.occupation || "",
      education: profile?.education || "",
    },
  });
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const catalog = useTraitCatalog();
  const [selected, setSelected] = useState<string[]>(profile?.traits ?? []);
  const [goals, setGoals] = useState<string[]>(profile?.datingGoals ?? []);
  const toggle = (
    list: string[],
    set: (next: string[]) => void,
    code: string,
  ) =>
    set(list.includes(code) ? list.filter((c) => c !== code) : [...list, code]);
  // 從其他管道匯入的檔案可能使用清單外的城市，保留它原本的座標。
  const cityOptions =
    profile && !cities.some((c) => c.name === profile.city)
      ? [
          ...cities,
          { name: profile.city, lat: profile.latitude, lng: profile.longitude },
        ]
      : cities;
  const save = form.handleSubmit(async (values) => {
    setError("");
    const city = cityOptions.find((c) => c.name === values.city);
    if (!city) {
      setError("請選擇居住城市。");
      return;
    }
    // 第一次建檔要把自我介紹與兩組選擇都填好，其他頁面才會解鎖。
    if (onboarding) {
      if (!values.bio.trim()) {
        setError("請寫一段自我介紹。");
        return;
      }
      if (!goals.length) {
        setError("請選擇想遇見的關係。");
        return;
      }
      if (!selected.length) {
        setError("請至少選一個小熱愛。");
        return;
      }
    }
    try {
      await send(
        "/profile",
        {
          ...values,
          heightCm: values.heightCm ? Number(values.heightCm) : null,
          latitude: city.lat,
          longitude: city.lng,
          traits: selected,
          datingGoals: goals,
        },
        "PUT",
      );
      await client.invalidateQueries({ queryKey: ["/profile"] });
      await client.invalidateQueries({ queryKey: ["/discovery"] });
      onSaved(
        onboarding
          ? "個人檔案已儲存，接著加入一張生活照。"
          : "個人檔案已儲存，現在可以開始探索。",
      );
    } catch (e) {
      setError((e as Error).message);
    }
  });
  async function upload(file?: File) {
    if (!file) return;
    if (file.size > photoLimit) {
      setError(photoTooLarge);
      return;
    }
    setBusy(true);
    setError("");
    try {
      const data = new FormData();
      data.append("file", file);
      await api("/profile/photos", { method: "POST", body: data });
      await client.invalidateQueries({ queryKey: ["/profile"] });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="form-layout">
      <section className="panel">
        <form onSubmit={save}>
          <div className="panel-title">
            <span className="eyebrow">
              {profile ? "EDIT PROFILE" : "CREATE PROFILE"}
            </span>
            <h2>{profile ? "編輯個人檔案" : "建立你的個人檔案"}</h2>
          </div>
          <div className="form-grid">
            <label>
              顯示名稱
              <input
                {...form.register("displayName")}
                maxLength={40}
                required
                placeholder="希望大家怎麼稱呼你？"
              />
            </label>
            <label>
              生日
              <input type="date" {...form.register("birthDate")} required />
              <small>須年滿 18 歲，只會公開年齡。</small>
            </label>
            <label>
              性別
              <select {...form.register("gender")}>
                {Object.entries(genderLabels)
                  .filter(([k]) => k !== "any")
                  .map(([k, v]) => (
                    <option key={k} value={k}>
                      {v}
                    </option>
                  ))}
              </select>
            </label>
            <label>
              居住城市
              <select {...form.register("city")}>
                {cityOptions.map((c) => (
                  <option key={c.name}>{c.name}</option>
                ))}
              </select>
              <small>使用城市中心估算距離，不公開精確位置。</small>
            </label>
            <label>
              身高（選填）
              <input
                type="number"
                inputMode="numeric"
                min={100}
                max={250}
                placeholder="公分"
                {...form.register("heightCm")}
              />
              <small>填了才會出現在別人的身高篩選結果裡。</small>
            </label>
            <label>
              職業（選填）
              <input {...form.register("occupation")} maxLength={80} />
            </label>
            <label>
              學歷（選填）
              <input {...form.register("education")} maxLength={80} />
            </label>
            <label className="span-two">
              自我介紹
              <textarea
                {...form.register("bio")}
                rows={4}
                maxLength={1000}
                placeholder="最近讓你開心的小事是什麼？"
              />
            </label>
          </div>
          <div className="divider" />
          <h3>想遇見的關係</h3>
          <p className="muted">最多選 {datingGoalLimit} 項。</p>
          <div className="tags selectable">
            {traitsOf(catalog, "dating_goal").map((goal) => {
              const on = goals.includes(goal.code);
              return (
                <button
                  type="button"
                  key={goal.code}
                  aria-pressed={on}
                  disabled={!on && goals.length >= datingGoalLimit}
                  onClick={() => toggle(goals, setGoals, goal.code)}
                >
                  {on && <Check size={14} />} {goal.label}
                </button>
              );
            })}
          </div>
          <div className="divider" />
          <h3>我的小熱愛</h3>
          <p className="muted">選擇你喜歡的事，為對話留一個起點。</p>
          {traitCategories(catalog).map((category) => (
            <fieldset className="tag-field" key={category}>
              <legend>{categoryTitle(category)}</legend>
              <div className="tags selectable">
                {traitsOf(catalog, category).map((trait) => {
                  const on = selected.includes(trait.code);
                  return (
                    <button
                      type="button"
                      key={trait.code}
                      aria-pressed={on}
                      onClick={() => toggle(selected, setSelected, trait.code)}
                    >
                      {on && <Check size={14} />} {trait.label}
                    </button>
                  );
                })}
              </div>
            </fieldset>
          ))}
          <ErrorText message={error || loadError} />
          <div className="form-actions">
            <button className="button" disabled={form.formState.isSubmitting}>
              {form.formState.isSubmitting ? "儲存中…" : "儲存個人檔案"}
              <Check size={17} />
            </button>
            {onCancel && (
              <button
                type="button"
                className="button secondary"
                onClick={onCancel}
              >
                取消
              </button>
            )}
          </div>
        </form>
      </section>
      <aside>
        <section className="panel photos-panel">
          <h2>我的生活照</h2>
          <p className="muted">
            {profile
              ? "第一張照片會成為你的主照片。最多 6 張，每張 8 MB。"
              : "先儲存「關於我」，就能加入照片。"}
          </p>
          <div className="photo-grid">
            {profile?.photos.map((photo) => (
              <div className="photo-tile" key={photo.id}>
                <img src={photo.url} alt="我的照片" />
                <button
                  type="button"
                  className="photo-delete"
                  aria-label="刪除照片"
                  onClick={async () => {
                    if (!window.confirm("確定刪除這張照片？")) return;
                    try {
                      await api(`/profile/photos/${photo.id}`, {
                        method: "DELETE",
                      });
                      await client.invalidateQueries({
                        queryKey: ["/profile"],
                      });
                    } catch (e) {
                      setError((e as Error).message);
                    }
                  }}
                >
                  <X size={16} />
                </button>
              </div>
            ))}
            {(profile?.photos.length || 0) < 6 && (
              <label
                className={profile ? "photo-upload" : "photo-upload disabled"}
              >
                <Camera size={23} />
                <span>
                  {busy ? "上傳中…" : profile ? "加入照片" : "先儲存檔案"}
                </span>
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  disabled={busy || !profile}
                  onChange={(e) => {
                    void upload(e.target.files?.[0]);
                    e.target.value = "";
                  }}
                />
              </label>
            )}
          </div>
        </section>
        <section className="panel compact">
          <ShieldCheck />
          <h3>讓真實，多一份安心。</h3>
          <p className="muted">
            查看你的驗證狀態。只有完成真人驗證，才會顯示驗證標記。
          </p>
          <Link href="/verification" className="text-link">
            查看真人驗證
            <ArrowRight size={16} />
          </Link>
        </section>
      </aside>
    </div>
  );
}
function PreferencesPage() {
  const query = useData<Preferences>("/preferences");
  const catalog = useTraitCatalog();
  const blocks =
    useData<{ blockedUserId: string; displayName: string }[]>("/blocks");
  const client = useQueryClient();
  const [values, setValues] = useState<Preferences>({
    minAge: 18,
    maxAge: 99,
    preferredGender: "any",
    maxDistanceKm: 100,
    minHeightCm: 130,
    maxHeightCm: 250,
    preferredDatingIntent: "any",
  });
  const [error, setError] = useState("");
  const [saved, setSaved] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  // 兩支拉桿代表同一個範圍：動其中一支時用最新狀態把另一支夾住，拖過頭也不會反過來。
  const setAges = (which: "low" | "high", value: number) =>
    setValues((v) =>
      which === "low"
        ? { ...v, minAge: value, maxAge: Math.max(v.maxAge, value) }
        : { ...v, minAge: Math.min(v.minAge, value), maxAge: value },
    );
  const setHeights = (which: "low" | "high", value: number) =>
    setValues((v) =>
      which === "low"
        ? {
            ...v,
            minHeightCm: value,
            maxHeightCm: Math.max(v.maxHeightCm, value),
          }
        : {
            ...v,
            minHeightCm: Math.min(v.minHeightCm, value),
            maxHeightCm: value,
          },
    );
  useEffect(() => {
    if (query.data) {
      const {
        minAge,
        maxAge,
        preferredGender,
        maxDistanceKm,
        minHeightCm,
        maxHeightCm,
        preferredDatingIntent,
      } = query.data;
      setValues({
        minAge,
        maxAge,
        preferredGender,
        maxDistanceKm,
        minHeightCm,
        maxHeightCm,
        preferredDatingIntent,
      });
      setLoaded(true);
    }
  }, [query.data]);
  if (!loaded)
    return query.error ? (
      <ErrorText message={query.error.message} />
    ) : (
      <Loading />
    );
  return (
    <>
      <Heading
        overline="FIND YOUR OWN RHYTHM"
        title="探索偏好"
        text="你的期待，值得被理解。探索會同時考量你與對方的偏好。"
      />
      <section className="panel narrow">
        <form
          onSubmit={async (e) => {
            e.preventDefault();
            setBusy(true);
            setError("");
            setSaved(false);
            try {
              await send("/preferences", values, "PUT");
              await client.invalidateQueries({ queryKey: ["/preferences"] });
              await client.invalidateQueries({ queryKey: ["/discovery"] });
              setSaved(true);
            } catch (e) {
              setError((e as Error).message);
            } finally {
              setBusy(false);
            }
          }}
        >
          <h2>我希望認識的人</h2>
          <div className="form-grid">
            <label className="span-two">
              年齡範圍：{values.minAge}–{values.maxAge} 歲
              <RangePair
                range={ageRange}
                low={values.minAge}
                high={values.maxAge}
                labels={["最小年齡", "最大年齡"]}
                onChange={setAges}
              />
            </label>
            <label className="span-two">
              身高範圍：{values.minHeightCm}–{values.maxHeightCm} 公分
              <RangePair
                range={heightRange}
                low={values.minHeightCm}
                high={values.maxHeightCm}
                labels={["最低身高", "最高身高"]}
                onChange={setHeights}
              />
              <small>沒有填身高的人仍會出現。</small>
            </label>
            <label>
              性別偏好
              <select
                value={values.preferredGender}
                onChange={(e) =>
                  setValues({ ...values, preferredGender: e.target.value })
                }
              >
                {Object.entries(genderLabels).map(([k, v]) => (
                  <option key={k} value={k}>
                    {v}
                  </option>
                ))}
              </select>
            </label>
            <label>
              關係期待
              <select
                value={values.preferredDatingIntent}
                onChange={(e) =>
                  setValues({
                    ...values,
                    preferredDatingIntent: e.target.value,
                  })
                }
              >
                <option value="any">都可以</option>
                {traitsOf(catalog, "dating_goal").map((goal) => (
                  <option key={goal.code} value={goal.code}>
                    {goal.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="span-two">
              探索距離：{values.maxDistanceKm} 公里
              <input
                type="range"
                className="range-single"
                style={rangeStyle(
                  distanceRange,
                  distanceRange[0],
                  Math.min(values.maxDistanceKm, distanceRange[1]),
                )}
                min={distanceRange[0]}
                max={distanceRange[1]}
                value={Math.min(values.maxDistanceKm, distanceRange[1])}
                onChange={(e) =>
                  setValues({
                    ...values,
                    maxDistanceKm: Number(e.target.value),
                  })
                }
              />
              <small>依城市中心估算，實際距離可能不同。</small>
            </label>
          </div>
          <ErrorText message={error || query.error?.message} />
          {saved && (
            <p className="success" role="status">
              探索偏好已更新。
            </p>
          )}
          <button className="button" disabled={busy}>
            {busy ? "儲存中…" : "儲存探索偏好"}
            <Check size={17} />
          </button>
        </form>
      </section>
      <section className="panel narrow">
        <h2>封鎖名單</h2>
        <p className="muted">解除封鎖不會恢復已結束的配對。</p>
        {blocks.data?.length ? (
          blocks.data.map((b) => (
            <div className="list-row" key={b.blockedUserId}>
              <span>{b.displayName}</span>
              <button
                className="button secondary small"
                onClick={async () => {
                  try {
                    await api(`/blocks/${b.blockedUserId}`, {
                      method: "DELETE",
                    });
                    await client.invalidateQueries({ queryKey: ["/blocks"] });
                  } catch (e) {
                    setError((e as Error).message);
                  }
                }}
              >
                解除封鎖
              </button>
            </div>
          ))
        ) : (
          <p>目前沒有封鎖任何人。</p>
        )}
      </section>
    </>
  );
}
function VerificationPage() {
  const q = useData<{
    status: string;
    reasonCode?: string;
    modelName?: string;
    createdAt?: string;
  }>("/verification/status");
  const client = useQueryClient();
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const labels: Record<string, string> = {
    not_started: "尚未驗證",
    pending: "驗證處理中",
    verified: "已通過驗證",
    rejected: "未通過驗證",
    unavailable: "驗證服務尚未就緒",
  };
  return (
    <>
      <Heading
        overline="TRUST STARTS WITH HONESTY"
        title="真人驗證"
        text="多一點真實，多一份安心。真人驗證包含身分比對與活體判斷，上傳照片本身不代表通過驗證。"
      />
      <section className="panel narrow">
        <div className="verification-status">
          <div className="empty-icon">
            <ShieldCheck size={34} />
          </div>
          <span className="eyebrow">YOUR VERIFICATION</span>
          <h2>{labels[q.data?.status || "not_started"]}</h2>
          {q.data?.status === "unavailable" && (
            <p className="muted">
              目前尚未接入可用的真人驗證模型，你的帳號仍為未驗證。
            </p>
          )}
          {q.data?.status === "rejected" && (
            <p className="muted">請確認照片清晰且為本人，然後重新上傳。</p>
          )}
          {q.data?.status === "verified" && (
            <p className="success">你的驗證已完成。</p>
          )}
        </div>
        <div className="divider" />
        <form
          onSubmit={async (e) => {
            e.preventDefault();
            if (!file) return;
            setBusy(true);
            setError("");
            try {
              const data = new FormData();
              data.append("file", file);
              await api("/onboarding/selfie", { method: "POST", body: data });
              await client.invalidateQueries({
                queryKey: ["/verification/status"],
              });
              const me = await api<User>("/auth/me");
              useAuth.getState().set(useAuth.getState().token, me);
            } catch (e) {
              setError((e as Error).message);
            } finally {
              setBusy(false);
            }
          }}
        >
          <label>
            選擇清晰的本人自拍
            <input
              required
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={(e) => {
                const picked = e.target.files?.[0] || null;
                if (picked && picked.size > photoLimit) {
                  setError(photoTooLarge);
                  setFile(null);
                  e.target.value = "";
                  return;
                }
                setError("");
                setFile(picked);
              }}
            />
            <small>JPEG、PNG、WebP，最大 8 MB；不會顯示在公開檔案。</small>
          </label>
          <p className="muted">
            此開發版本僅在本次請求中處理自拍，不保存原始影像。正式身分驗證與資料政策尚待設定。
          </p>
          <label className="check-label">
            <input type="checkbox" required />
            我同意使用這張自拍進行本次驗證處理。
          </label>
          <ErrorText message={error || q.error?.message} />
          <button className="button" disabled={busy || !file}>
            {busy ? "處理中…" : "提交驗證"}
            <ShieldCheck size={18} />
          </button>
        </form>
      </section>
    </>
  );
}
const matchTags = [
  { key: "matched", label: "已配對" },
  { key: "liked", label: "我喜歡的" },
] as const;
type MatchTag = (typeof matchTags)[number]["key"];
function MatchesPage() {
  const q = useData<Match[]>("/matches");
  const likes = useData<SentLike[]>("/likes");
  const client = useQueryClient();
  const [error, setError] = useState("");
  const [tag, setTag] = useState<MatchTag>("matched");
  const [person, setPerson] = useState<Card | null>(null);
  const matches = q.data ?? [];
  // 已配對以 /matches 為準；兩個查詢刷新有先後，這裡再排除一次，同一個人才不會出現兩張卡。
  const matchedIds = new Set(matches.map((m) => m.otherUser.userId));
  const liked = (likes.data ?? []).filter(
    (l) => l.status === "waiting" && !matchedIds.has(l.targetUserId),
  );
  const counts: Record<MatchTag, number> = {
    matched: matches.length,
    liked: liked.length,
  };
  const total = counts.matched + counts.liked;
  return (
    <>
      <Heading
        overline="BETTER TOGETHER"
        title="我的配對"
        text="相互喜歡，是故事的第一頁。"
      >
        {!!total && (
          <div className="section-bar">
            <div className="tags selectable" role="group" aria-label="篩選">
              {matchTags.map((t) => (
                <button
                  type="button"
                  key={t.key}
                  aria-pressed={tag === t.key}
                  onClick={() => setTag(t.key)}
                >
                  {t.label} {counts[t.key]}
                </button>
              ))}
            </div>
            <Link className="button secondary" href="/preferences">
              探索偏好
            </Link>
          </div>
        )}
      </Heading>
      <ErrorText message={error || q.error?.message || likes.error?.message} />
      {q.isLoading || likes.isLoading ? (
        <Loading />
      ) : !total ? (
        <Empty
          title="你的下一個火花，還在路上"
          text="到探索看看，彼此喜歡後會在這裡相遇。"
          link="/discover"
          label="開始探索"
        />
      ) : !counts[tag] ? (
        tag === "matched" ? (
          <Empty
            title="還沒有互相喜歡的人"
            text="對方也按下喜歡後，就會出現在這裡。"
          />
        ) : (
          <Empty
            title="目前沒有等待回應的喜歡"
            text="在探索按下喜歡的人，會先出現在這裡。"
            link="/discover"
            label="開始探索"
          />
        )
      ) : (
        <div className="match-grid">
          {tag === "matched" &&
            matches.map((m) => {
              const fresh = matchedLabel(m.createdAt);
              return (
                <article
                  className={fresh ? "match-card fresh" : "match-card"}
                  key={m.id}
                >
                  <Portrait person={m.otherUser} large />
                  <div className="match-copy">
                    <p className="match-flag">
                      <span className="badge">已配對</span>
                      {fresh}
                    </p>
                    <h2>
                      {m.otherUser.displayName} <span>{m.otherUser.age}</span>
                    </h2>
                    <p>
                      {m.otherUser.bio ||
                        [m.otherUser.city, goalText(m.otherUser.datingGoals)]
                          .filter(Boolean)
                          .join(" · ")}
                    </p>
                    <div className="match-actions">
                      <Link
                        className="button"
                        href={`/messages/${m.conversationId}`}
                      >
                        開始聊天
                      </Link>
                      <div className="sub-actions">
                        <button
                          onClick={async () => {
                            if (
                              !window.confirm(
                                "確定結束配對？這段對話將無法繼續。",
                              )
                            )
                              return;
                            try {
                              await api(`/matches/${m.id}`, {
                                method: "DELETE",
                              });
                              await client.invalidateQueries();
                            } catch (e) {
                              setError((e as Error).message);
                            }
                          }}
                        >
                          結束配對
                        </button>
                        <button
                          onClick={async () => {
                            if (!window.confirm("確定封鎖對方並結束配對？"))
                              return;
                            try {
                              await send("/blocks", {
                                blockedUserId: m.otherUser.userId,
                              });
                              await client.invalidateQueries();
                            } catch (e) {
                              setError((e as Error).message);
                            }
                          }}
                        >
                          封鎖
                        </button>
                      </div>
                    </div>
                  </div>
                </article>
              );
            })}
          {tag === "liked" &&
            liked.map((l) => (
              <article className="match-card" key={l.targetUserId}>
                <Portrait person={l.user} large />
                <div className="match-copy">
                  <p className="match-flag">
                    <span className="badge soft">我喜歡的</span>
                    {likedLabel(l.createdAt)}
                  </p>
                  <h2>
                    {l.user.displayName} <span>{l.user.age}</span>
                  </h2>
                  <p>
                    {l.user.bio ||
                      [l.user.city, goalText(l.user.datingGoals)]
                        .filter(Boolean)
                        .join(" · ")}
                  </p>
                  <div className="match-actions">
                    <button
                      type="button"
                      className="button secondary"
                      onClick={() => setPerson(l.user)}
                    >
                      看看檔案
                    </button>
                  </div>
                </div>
              </article>
            ))}
        </div>
      )}
      {person && (
        <PersonDialog person={person} onClose={() => setPerson(null)} />
      )}
    </>
  );
}
function MessagesPage({ id, socket }: { id?: string; socket: Socket | null }) {
  const q = useData<Conversation[]>("/conversations");
  // 對話列表要顯示每個人的上線狀態：presence 事件只有狀態變化時才會來，
  // 所以連上線時先跟伺服器要一份「現在誰在線上」的快照。
  const [onlineIds, setOnlineIds] = useState<string[]>([]);
  useEffect(() => {
    if (!socket) return;
    const snapshot = () =>
      socket.emit(
        "presence:list",
        {},
        (ack: { ok: boolean; online?: string[] }) => {
          if (ack.ok && ack.online) setOnlineIds(ack.online);
        },
      );
    const update = (d: { userId: string; online: boolean }) =>
      setOnlineIds((ids) =>
        d.online
          ? ids.includes(d.userId)
            ? ids
            : [...ids, d.userId]
          : ids.filter((userId) => userId !== d.userId),
      );
    snapshot();
    socket.on("connect", snapshot);
    socket.on("presence", update);
    return () => {
      socket.off("connect", snapshot);
      socket.off("presence", update);
    };
  }, [socket]);
  const index = q.data?.findIndex((c) => c.id === id) ?? -1;
  const selected = index >= 0 ? q.data?.[index] : undefined;
  const unread = q.data?.reduce((sum, c) => sum + c.unreadCount, 0) || 0;
  return (
    <>
      <Heading
        overline="LET'S TALK"
        title="聊天室"
        text="一句你好，也許就是故事的開始。"
      />
      <ErrorText message={q.error?.message} />
      {q.isLoading ? (
        <Loading />
      ) : !q.data?.length ? (
        <Empty
          title="還沒有對話"
          text="完成雙向配對後，就能開始聊天。"
          link="/discover"
          label="探索新朋友"
        />
      ) : (
        <>
          {id && !selected && (
            <p className="muted chat-notice" role="status">
              這段對話已結束或不存在，請從列表選擇其他對話。
            </p>
          )}
          {/* 找不到對話時不能切成手機的單欄聊天畫面，否則列表與提示都被隱藏，只剩空白 */}
          <div className={`chat-layout ${selected ? "has-selection" : ""}`}>
            <aside className="conversation-list">
              <div className="list-heading">
                <h2>最近對話</h2>
                <p>{unread} 則未讀訊息</p>
              </div>
              <div className="conversation-items">
                {q.data.map((c) => (
                  <Link
                    href={`/messages/${c.id}`}
                    className={
                      id === c.id ? "conversation selected" : "conversation"
                    }
                    key={c.id}
                  >
                    <span className="conversation-avatar">
                      <Portrait person={c.otherUser} />
                      {/* 綠燈只在對方上線時出現。 */}
                      {onlineIds.includes(c.otherUser.userId) && (
                        <span className="online-dot" aria-label="上線中" />
                      )}
                    </span>
                    <div className="conversation-text">
                      <div className="conversation-top">
                        <b>{c.otherUser.displayName}</b>
                      </div>
                      <p>{c.lastMessage?.content || "從一句你好開始吧。"}</p>
                    </div>
                    {/* 時間與未讀數放同一欄並置中，右側才會對齊。 */}
                    <div className="conversation-meta">
                      <time>{listTime(c.lastMessage?.createdAt)}</time>
                      {c.unreadCount > 0 && (
                        <span className="unread">{c.unreadCount}</span>
                      )}
                    </div>
                  </Link>
                ))}
              </div>
              <Link className="button secondary list-more" href="/matches">
                查看所有配對
              </Link>
            </aside>
            {selected ? (
              <Chat key={selected.id} conversation={selected} socket={socket} />
            ) : (
              <div className="chat-placeholder">
                <MessageCircle size={42} />
                <h2>選一段對話，聊聊今天。</h2>
              </div>
            )}
          </div>
        </>
      )}
    </>
  );
}
function Chat({
  conversation: c,
  socket,
}: {
  conversation: Conversation;
  socket: Socket | null;
}) {
  const user = useAuth((s) => s.user)!;
  const otherId = c.otherUser.userId;
  const q = useData<Message[]>(`/conversations/${c.id}/messages`);
  const client = useQueryClient();
  const [messages, setMessages] = useState<Message[]>([]);
  const [content, setContent] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [typing, setTyping] = useState(false);
  const [online, setOnline] = useState(false);
  const [readAt, setReadAt] = useState(c.otherLastReadAt || "");
  const [closed, setClosed] = useState(false);
  const [hasOlder, setHasOlder] = useState(true);
  const [showProfile, setShowProfile] = useState(false);
  const end = useRef<HTMLDivElement>(null);
  const draft = useRef<{
    content: string;
    clientId: string;
    suggestionId?: string;
  } | null>(null);
  const typingTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastTyping = useRef(0);
  // ── AI 推薦回覆（docs/ai/REPLY-SUGGESTIONS-SPEC.md）──────────────────
  // 第 1 則會打字填進輸入框，這裡放的是「其餘建議」，顯示成輸入框上方的按鈕。
  const [suggestions, setSuggestions] = useState<ReplySuggestion[]>([]);
  // 等後端回應的期間：輸入框邊框跑彩光，AI 鈕與傳送鍵都停用。
  const [suggesting, setSuggesting] = useState(false);
  // 不足 3 則時後端會附一句說明（規格 4.1：顯示剩下的就好，不硬湊）。
  const [notice, setNotice] = useState("");
  // 目前輸入框的內容來自哪一則推薦；送出時一起帶給後端判斷訊息來源（規格 5.6）。
  const suggestionId = useRef<string | null>(null);
  // 打字動畫的計時器；使用者自己打字、送出或離開聊天室時都要清掉。
  const typer = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => {
    const saved = c.otherLastReadAt;
    if (saved) setReadAt((current) => (current > saved ? current : saved));
  }, [c.otherLastReadAt]);
  const merge = (items: Message[]) =>
    setMessages((old) =>
      Array.from(
        new Map([...old, ...items].map((m) => [m.id, m])).values(),
      ).sort((a, b) =>
        a.createdAt < b.createdAt
          ? -1
          : a.createdAt > b.createdAt
            ? 1
            : a.id < b.id
              ? -1
              : a.id > b.id
                ? 1
                : 0,
      ),
    );
  useEffect(() => {
    if (q.data) {
      merge(q.data);
      setHasOlder(q.data.length === 50);
    }
  }, [q.data]);
  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [messages.length]);
  // 切換聊天室或離開頁面時停掉打字動畫，計時器不會在元件消失後繼續跑。
  useEffect(() => () => stopTyping(), [c.id]);
  useEffect(() => {
    const markRead = () => {
      if (document.visibilityState === "visible")
        void send(`/conversations/${c.id}/read`, {})
          .then(() =>
            client.invalidateQueries({ queryKey: ["/conversations"] }),
          )
          .catch(() => {});
    };
    markRead();
    document.addEventListener("visibilitychange", markRead);
    if (!socket)
      return () => document.removeEventListener("visibilitychange", markRead);
    const join = () =>
      socket.emit(
        "conversation:join",
        { conversationId: c.id },
        (ack: { ok: boolean }) => {
          if (!ack.ok) setClosed(true);
          else
            void client.invalidateQueries({
              queryKey: [`/conversations/${c.id}/messages`],
            });
        },
      );
    join();
    socket.on("connect", join);
    const receive = (m: Message) => {
      if (m.conversationId === c.id) {
        merge([m]);
        markRead();
      }
    };
    // 同一位使用者開多個分頁時，也會收到自己的輸入中與上線事件，只採用對方的。
    const type = (d: {
      conversationId: string;
      isTyping: boolean;
      userId: string;
    }) => {
      if (d.conversationId === c.id && d.userId === otherId) {
        setTyping(d.isTyping);
        if (typingTimer.current) clearTimeout(typingTimer.current);
        typingTimer.current = setTimeout(() => setTyping(false), 3500);
      }
    };
    const presence = (d: {
      conversationId: string;
      online: boolean;
      userId: string;
    }) => {
      if (d.conversationId === c.id && d.userId === otherId)
        setOnline(d.online);
    };
    const read = (d: { conversationId: string; readAt: string }) => {
      if (d.conversationId === c.id) setReadAt(d.readAt);
    };
    const close = (d: { conversationId: string }) => {
      if (d.conversationId === c.id) setClosed(true);
    };
    socket.on("message:new", receive);
    socket.on("typing", type);
    socket.on("presence", presence);
    socket.on("conversation:read", read);
    socket.on("conversation:closed", close);
    return () => {
      document.removeEventListener("visibilitychange", markRead);
      socket.off("connect", join);
      socket.off("message:new", receive);
      socket.off("typing", type);
      socket.off("presence", presence);
      socket.off("conversation:read", read);
      socket.off("conversation:closed", close);
      if (typingTimer.current) clearTimeout(typingTimer.current);
    };
  }, [c.id, otherId, socket, client]);
  /**
   * 停掉打字動畫。
   * 使用者自己打字、送出訊息或離開聊天室時都要呼叫；
   * 不停掉的話計時器會繼續往輸入框塞字，把使用者打到一半的內容蓋掉。
   */
  function stopTyping() {
    if (typer.current) clearInterval(typer.current);
    typer.current = null;
  }
  /**
   * 把第 1 則推薦用打字動畫填進輸入框（每 40 毫秒一個字）。
   *
   * 同時記下這則推薦的 id：即使使用者之後改了幾個字，送出時仍然會帶著它，
   * 由後端比對相似度決定 ai_verbatim／ai_edited／human（規格 5.6）。
   */
  function typeIn(text: string, id: string) {
    stopTyping();
    suggestionId.current = id;
    setContent("");
    let shown = 0;
    typer.current = setInterval(() => {
      shown += 1;
      setContent(text.slice(0, shown));
      if (shown >= text.length) stopTyping();
    }, 40);
  }
  /**
   * 點了輸入框上方的建議按鈕：直接換掉輸入框內容（不跑打字動畫），
   * 並把來源換成這一則推薦。
   */
  function applySuggestion(suggestion: ReplySuggestion) {
    stopTyping();
    suggestionId.current = suggestion.id;
    setContent(suggestion.text);
  }
  /**
   * 按下輸入框裡的「AI 推薦」：向後端要一批建議。
   *
   * 成功：第 1 則打字填入輸入框，其餘變成上方的按鈕；按鈕會變成「換一批」，
   * 再按一次後端會避開同一情境下已經給過的句子。
   * 失敗：顯示後端回來的中文訊息（例如「AI 忙碌中，請稍後再試。」），輸入框內容不動。
   */
  async function askAi() {
    if (suggesting || closed) return;
    setSuggesting(true);
    setError("");
    setNotice("");
    try {
      const result = await requestSuggestions(c.id);
      const [first, ...rest] = result.suggestions;
      setSuggestions(rest);
      setNotice(result.notice || "");
      if (first) typeIn(first.text, first.id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSuggesting(false);
    }
  }
  async function submit(e: React.FormEvent) {
    e.preventDefault();
    // 等待 AI 推薦時傳送鍵是停用的；按 Enter 也會走到這裡，所以要一起擋下。
    if (!content.trim() || busy || suggesting || closed) return;
    setBusy(true);
    setError("");
    if (!draft.current || draft.current.content !== content.trim())
      draft.current = {
        content: content.trim(),
        clientId: crypto.randomUUID(),
        // 這則訊息是從哪一則 AI 推薦來的；沒有用推薦就不送這個欄位。
        ...(suggestionId.current ? { suggestionId: suggestionId.current } : {}),
      };
    try {
      const message = await send<Message>(
        `/conversations/${c.id}/messages`,
        draft.current,
      );
      merge([message]);
      setContent("");
      draft.current = null;
      // 送出後收起這一批建議：對話已經往前走，舊建議不再適用（要新的就再按一次）。
      stopTyping();
      suggestionId.current = null;
      setSuggestions([]);
      setNotice("");
      socket?.emit("typing", { conversationId: c.id, isTyping: false });
      await client.invalidateQueries({ queryKey: ["/conversations"] });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="chat-panel">
      <header className="chat-header">
        <Link
          className="mobile-back"
          href="/messages"
          aria-label="返回對話列表"
        >
          <ChevronLeft />
        </Link>
        <div className="chat-title">
          <h2>{c.otherUser.displayName}</h2>
          <small className={online ? "presence online" : "presence"}>
            {online ? "上線" : "離線"}
          </small>
        </div>
        <button
          type="button"
          className="button secondary"
          onClick={() => setShowProfile(true)}
        >
          查看檔案
        </button>
      </header>
      <div className="message-scroll">
        {hasOlder && messages[0] && (
          <button
            className="load-older"
            onClick={async () => {
              try {
                const oldest = messages[0];
                const older = await api<Message[]>(
                  `/conversations/${c.id}/messages?before=${encodeURIComponent(oldest.createdAt)}&beforeId=${oldest.id}`,
                );
                merge(older);
                setHasOlder(older.length === 50);
              } catch (e) {
                setError((e as Error).message);
              }
            }}
          >
            載入較早訊息
          </button>
        )}
        {q.isLoading ? (
          <Loading />
        ) : messages.length === 0 ? (
          <div className="conversation-start">
            <HeartIcon size={26} />
            <h3>你們的故事，從這裡開始。</h3>
            <p>「你最近最喜歡的一間咖啡店是哪間？」</p>
          </div>
        ) : (
          messages.map((m, i) => {
            const own = m.senderId === user.id;
            const newDay =
              i === 0 ||
              daysAgo(messages[i - 1].createdAt) !== daysAgo(m.createdAt);
            return (
              <Fragment key={m.id}>
                {newDay && (
                  <p className="date-chip">
                    <span>{dayLabel(m.createdAt)}</span>
                  </p>
                )}
                <div className={own ? "message own" : "message"}>
                  {!own && <Portrait person={c.otherUser} small />}
                  <div className="message-body">
                    <div className="bubble">{m.content}</div>
                    <small>
                      {clock(m.createdAt)}
                      {/* 勾勾與「已讀」包成一行，顯示在送出時間的下面。 */}
                      {own && readAt >= m.createdAt && (
                        <span className="read-mark">
                          <CheckCheck size={13} />
                          已讀
                        </span>
                      )}
                    </small>
                  </div>
                </div>
              </Fragment>
            );
          })
        )}
        <div ref={end} />
      </div>
      <ErrorText message={error || q.error?.message} />
      <form className="message-input" onSubmit={submit}>
        <div className="typing-line" aria-live="polite">
          {typing ? "對方正在輸入…" : closed ? "這段對話已結束。" : ""}
        </div>
        {(suggestions.length > 0 || notice) && (
          <div className="ai-chips" aria-live="polite">
            {suggestions.map((s) => (
              // 點一下就把這一則換進輸入框；title 讓太長被截斷時仍看得到全文。
              <button
                key={s.id}
                type="button"
                title={s.text}
                onClick={() => applySuggestion(s)}
              >
                {s.text}
              </button>
            ))}
            {notice && <span className="ai-note">{notice}</span>}
          </div>
        )}
        <div className="composer-row">
          {/* 等待 AI 時多疊三層：光暈、彩光，以及蓋住中央的內底（做法同設計稿），
              所以看到的是一圈沿著邊框順時針繞行的光。 */}
          <div className={suggesting ? "ai-shell busy" : "ai-shell"}>
            {suggesting && (
              <>
                <span className="ai-glow" aria-hidden="true">
                  <i />
                </span>
                <span className="ai-ring" aria-hidden="true">
                  <i />
                </span>
                <span className="ai-fill" aria-hidden="true" />
              </>
            )}
            <input
              aria-label="訊息內容"
              placeholder="輸入訊息..."
              value={content}
              maxLength={2000}
              disabled={closed}
              onKeyDown={(e) => {
                // 按 Enter 直接送出。中文輸入法選字時的 Enter 是「確認選字」，
                // 那時 isComposing 為 true，不能當成送出，否則會把半形的注音送出去。
                if (
                  e.key !== "Enter" ||
                  e.shiftKey ||
                  e.nativeEvent.isComposing
                )
                  return;
                e.preventDefault();
                void submit(e);
              }}
              onChange={(e) => {
                // 使用者自己打字就中斷打字動畫；清空輸入框等於放棄這則推薦。
                stopTyping();
                if (!e.target.value) suggestionId.current = null;
                setContent(e.target.value);
                if (Date.now() - lastTyping.current > 1200) {
                  socket?.emit("typing", {
                    conversationId: c.id,
                    isTyping: !!e.target.value,
                  });
                  lastTyping.current = Date.now();
                }
              }}
            />
            {/* type="button"：它在 form 裡面，不加會變成送出訊息。 */}
            <button
              type="button"
              className={suggesting ? "ai-suggest busy" : "ai-suggest"}
              aria-label="AI 推薦回覆"
              disabled={suggesting || closed}
              onClick={askAi}
            >
              {suggesting ? (
                <Loader2 size={15} className="spin" />
              ) : suggestions.length ? (
                <RefreshCw size={15} />
              ) : (
                <Sparkles size={15} />
              )}
              {/* 手機版只留圖示，這段文字會被 CSS 收起來。 */}
              <span>
                {suggesting
                  ? "產生中"
                  : suggestions.length
                    ? "換一批"
                    : "AI 推薦"}
              </span>
            </button>
          </div>
          <button
            className="button"
            aria-label="傳送訊息"
            disabled={busy || suggesting || !content.trim() || closed}
          >
            {/* 手機版只留紙飛機圖示，桌面版維持「傳送」兩個字（由 CSS 切換）。 */}
            <Send className="send-icon" size={18} aria-hidden="true" />
            <span className="send-text">傳送</span>
          </button>
        </div>
      </form>
      {showProfile && (
        <PersonDialog
          person={c.otherUser}
          onClose={() => setShowProfile(false)}
        />
      )}
    </section>
  );
}
function NotificationsPage() {
  const q = useData<
    {
      id: string;
      type: string;
      payload: { conversationId?: string };
      readAt: string | null;
      createdAt: string;
    }[]
  >("/notifications");
  const client = useQueryClient();
  const router = useRouter();
  const [error, setError] = useState("");
  return (
    <>
      <Heading
        overline="LITTLE UPDATES, NEW POSSIBILITIES"
        title="通知"
        text="有人，正在靠近你的故事。在這裡查看新的配對與訊息。"
      />
      <ErrorText message={error || q.error?.message} />
      {q.isLoading ? (
        <Loading />
      ) : !q.data?.length ? (
        <Empty
          title="目前沒有新通知"
          text="當有人與你配對或傳來訊息，會在這裡提醒你。"
        />
      ) : (
        <section className="panel narrow">
          {q.data.map((n) => (
            <button
              className={`notification ${n.readAt ? "" : "new"}`}
              key={n.id}
              onClick={async () => {
                try {
                  await send(`/notifications/${n.id}/read`, {});
                  await client.invalidateQueries({
                    queryKey: ["/notifications"],
                  });
                  router.push(
                    n.payload.conversationId
                      ? `/messages/${n.payload.conversationId}`
                      : "/matches",
                  );
                } catch (e) {
                  setError((e as Error).message);
                }
              }}
            >
              <span className="notification-icon">
                {n.type === "match" ? (
                  <HeartIcon size={21} />
                ) : (
                  <MessageCircle size={21} />
                )}
              </span>
              <span>
                <b>
                  {n.type === "match" ? "你有新的雙向配對" : "你收到一則新訊息"}
                </b>
                <small>{new Date(n.createdAt).toLocaleString("zh-TW")}</small>
              </span>
              {!n.readAt && <i className="dot" />}
              <ArrowRight size={17} />
            </button>
          ))}
        </section>
      )}
    </>
  );
}
