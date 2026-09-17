"use client";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
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
  ChevronLeft,
  Compass,
  Heart,
  Loader2,
  LogOut,
  MapPin,
  MessageCircle,
  Send,
  Settings2,
  ShieldCheck,
  Sparkles,
  UserRound,
  X,
  Camera,
  CheckCheck,
} from "lucide-react";
import {
  api,
  refresh,
  send,
  useAuth,
  intentLabels,
  genderLabels,
  type Card,
  type Profile,
  type Preferences,
  type Match,
  type Conversation,
  type Message,
  type User,
} from "@/lib/api";
function useData<T>(path: string) {
  const user = useAuth((s) => s.user);
  return useQuery<T>({
    queryKey: [path, user?.id],
    queryFn: () => api<T>(path),
    enabled: !!user,
  });
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
      <p>正在準備你的遇見…</p>
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
}: {
  person: { displayName: string; photos?: { url: string }[] };
  large?: boolean;
}) {
  return (
    <div className={large ? "portrait large" : "portrait"}>
      {person.photos?.[0] ? (
        <img src={person.photos[0].url} alt={`${person.displayName}的照片`} />
      ) : (
        <span>{person.displayName.slice(0, 1)}</span>
      )}
    </div>
  );
}
const nav = [
  { href: "/discover", label: "探索", icon: Compass },
  { href: "/matches", label: "配對", icon: Heart },
  { href: "/messages", label: "訊息", icon: MessageCircle },
  { href: "/profile", label: "我的檔案", icon: UserRound },
];
export function DatingApp() {
  const path = usePathname();
  const router = useRouter();
  const { user, ready } = useAuth();
  const client = useQueryClient();
  const [socket, setSocket] = useState<Socket | null>(null);
  const token = useAuth((s) => s.token);
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
          ["/conversations", "/matches", "/notifications"].includes(
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
  const active = nav.find((n) => path.startsWith(n.href));
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Link href="/discover" className="brand">
          <span className="brand-mark">m</span>遇見
          <span className="brand-en">meet</span>
        </Link>
        <span className="eyebrow nav-label">YOUR NEXT CHAPTER</span>
        <nav>
          {nav.map((n) => (
            <Link
              key={n.href}
              href={n.href}
              className={
                path.startsWith(n.href) ? "nav-item active" : "nav-item"
              }
            >
              <n.icon size={22} />
              <span>{n.label}</span>
              {path.startsWith(n.href) && <i />}
            </Link>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="small-note">
            <Sparkles size={20} />
            <p>
              好的關係，
              <br />
              從做自己開始。
            </p>
          </div>
          <Link href="/preferences" className="nav-item">
            <Settings2 size={20} />
            探索偏好
          </Link>
          <Link href="/verification" className="nav-item">
            <ShieldCheck size={20} />
            真人驗證
          </Link>
          <button className="nav-item" onClick={logout}>
            <LogOut size={20} />
            登出
          </button>
        </div>
      </aside>
      <div className="workspace">
        <header className="topbar">
          <span className="breadcrumb">
            遇見 / <b>{active?.label || "帳號設定"}</b>
          </span>
          <div className="top-actions">
            <button
              className="icon-button mobile-logout"
              aria-label="登出"
              onClick={logout}
            >
              <LogOut size={20} />
            </button>
            <span className="local-badge">慢一點，也很好</span>
            <Link
              href="/notifications"
              aria-label="通知"
              className="icon-button"
            >
              <Bell size={20} />
            </Link>
            <Link
              href="/profile"
              className="mini-avatar"
              aria-label="我的個人資料"
            >
              {user.email.slice(0, 1).toUpperCase()}
            </Link>
          </div>
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
        <footer className="app-footer">
          MEET SOMEONE. BE YOURSELF.<span>讓每一次遇見，都有意義。</span>
        </footer>
      </div>
      <nav className="mobile-nav">
        {nav.map((n) => (
          <Link
            key={n.href}
            href={n.href}
            className={path.startsWith(n.href) ? "active" : ""}
          >
            <n.icon size={21} />
            {n.label}
          </Link>
        ))}
      </nav>
    </div>
  );
}
function Landing() {
  const user = useAuth((s) => s.user);
  return (
    <div className="landing">
      <header>
        <Link href="/" className="brand">
          <span className="brand-mark">m</span>遇見
          <span className="brand-en">meet</span>
        </Link>
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
          <Link
            className="button coral big"
            href={user ? "/discover" : "/register"}
          >
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
            <Heart size={58} strokeWidth={1.2} />
            <span>
              HELLO,
              <br />
              SOMETHING
              <br />
              <em>real.</em>
            </span>
          </div>
          <div className="floating-note">
            <Sparkles size={18} /> 為真實的相遇，留一點空間。
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
        遇見 meet <span>給關係一點可能，給自己一點時間。</span>
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
        <Link href="/" className="brand light">
          <span className="brand-mark">m</span>遇見 meet
        </Link>
        <div>
          <span className="eyebrow">A LITTLE HELLO GOES A LONG WAY</span>
          <h1>
            有些美好，
            <br />
            從一句你好開始。
          </h1>
          <div className="auth-flower">✳</div>
        </div>
        <small>每段故事，從真實開始。</small>
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
            {register ? "已經有帳號？" : "第一次來到遇見？"}
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
  children,
}: {
  overline: string;
  title: string;
  text: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="page-heading">
      <div>
        <span className="eyebrow">{overline}</span>
        <h1>{title}</h1>
        <p>{text}</p>
      </div>
      {children}
    </div>
  );
}
function Discover() {
  const query = useData<Card[]>("/discovery");
  const profile = useData<Profile | null>("/profile");
  const preferences = useData<Preferences>("/preferences");
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
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <Heading
        overline="DISCOVER YOUR KIND OF PERSON"
        title="今天，想遇見誰？"
        text="從共同喜好開始，讓故事自然發生。"
      >
        <Link className="button secondary" href="/preferences">
          <Settings2 size={17} />
          調整偏好
        </Link>
      </Heading>
      <div className="discover-layout">
        <section>
          <div className="section-label">
            <span>
              <i className="dot" /> 為你探索
            </span>
            <small>{query.data?.length || 0} 個可能的開始</small>
          </div>
          <ErrorText message={error || query.error?.message} />
          {match && (
            <div className="success" role="status">
              <Heart size={20} />
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
            <>
              <article className="discovery-card">
                <div className="discovery-image">
                  <Portrait person={person} large />
                  <span className="image-badge">
                    <MapPin size={14} />
                    {person.city}
                  </span>
                  <div className="image-caption">
                    <h2>
                      {person.displayName}
                      <span>{person.age}</span>
                      {person.isVerified && (
                        <ShieldCheck size={25} aria-label="已通過驗證" />
                      )}
                    </h2>
                    <p>{intentLabels[person.datingIntent]}</p>
                  </div>
                </div>
                <div className="card-copy">
                  <span className="eyebrow">A LITTLE ABOUT ME</span>
                  <p>{person.bio || "有些故事，適合在對話裡慢慢認識。"}</p>
                  <div className="tags">
                    {[
                      ...person.interests,
                      ...person.hobbies,
                      ...person.foods,
                    ].map((tag) => (
                      <span key={tag}>{tag}</span>
                    ))}
                  </div>
                </div>
              </article>
              <div className="discovery-actions">
                <button
                  className="round-button pass"
                  disabled={busy}
                  onClick={() => act("pass")}
                  aria-label="略過"
                >
                  <X size={26} />
                </button>
                <span>跟著感覺，慢慢來。</span>
                <button
                  className="round-button like"
                  disabled={busy}
                  onClick={() => act("like")}
                  aria-label="喜歡"
                >
                  <Heart size={27} />
                </button>
              </div>
            </>
          )}
        </section>
        <aside className="discover-aside">
          <div className="note-card">
            <Sparkles size={25} />
            <h3>
              不用很完美，
              <br />
              只要很真實。
            </h3>
            <p>比起精心設計的開場白，一個真誠的好奇，往往更動人。</p>
            <span>BE YOURSELF. ALWAYS.</span>
          </div>
          <div className="preference-card">
            <h3>你的探索指南</h3>
            <p>
              <MapPin size={17} />
              {profile.data?.city || "尚未設定城市"}附近
            </p>
            <p>
              <UserRound size={17} />
              {preferences.data
                ? `${preferences.data.minAge}–${preferences.data.maxAge} 歲 · ${genderLabels[preferences.data.preferredGender]}`
                : "讀取中…"}
            </p>
            <p>
              <Compass size={17} />
              {preferences.data?.maxDistanceKm || 100} 公里內
            </p>
            <Link href="/preferences">
              編輯偏好
              <ArrowUpRight size={16} />
            </Link>
          </div>
          <p className="muted footnote">
            <ShieldCheck size={16} />
            彼此喜歡才開啟聊天。你隨時可以封鎖或結束配對。
          </p>
        </aside>
      </div>
    </>
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
const profileFields = [
  "displayName",
  "birthDate",
  "gender",
  "bio",
  "city",
  "datingIntent",
  "occupation",
  "education",
] as const;
const tagGroups = ["interests", "hobbies", "foods"] as const;
type ProfileForm = Record<(typeof profileFields)[number], string>;
type Tags = Record<(typeof tagGroups)[number], string[]>;
const formOf = (p: Profile): ProfileForm => ({
  displayName: p.displayName,
  birthDate: p.birthDate,
  gender: p.gender,
  bio: p.bio,
  city: p.city,
  datingIntent: p.datingIntent,
  occupation: p.occupation || "",
  education: p.education || "",
});
const tagsOf = (p: Profile): Tags => ({
  interests: p.interests,
  hobbies: p.hobbies,
  foods: p.foods,
});
const sameTags = (a: string[], b: string[]) =>
  a.length === b.length && a.every((tag, i) => tag === b[i]);
const photoLimit = 8 * 1024 * 1024;
const photoTooLarge = "照片太大，請選擇 8 MB 以內的檔案。";
function ProfilePage() {
  const query = useData<Profile | null>("/profile");
  const client = useQueryClient();
  const form = useForm<ProfileForm>({
    defaultValues: {
      displayName: "",
      birthDate: "",
      gender: "woman",
      bio: "",
      city: "台北市",
      datingIntent: "serious",
      occupation: "",
      education: "",
    },
  });
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [busy, setBusy] = useState(false);
  const [tags, setTags] = useState<Tags>({
    interests: [],
    hobbies: [],
    foods: [],
  });
  const interests = useData<string[]>("/interests"),
    hobbies = useData<string[]>("/hobbies"),
    foods = useData<string[]>("/foods");
  // 表單上次同步的伺服器資料。重新取得資料（例如上傳照片後）時，只更新使用者之後沒改過的欄位，
  // 避免清掉尚未儲存的輸入。
  const synced = useRef<{ form: ProfileForm; tags: Tags } | null>(null);
  useEffect(() => {
    if (!query.data) return;
    const next = { form: formOf(query.data), tags: tagsOf(query.data) };
    const prev = synced.current;
    synced.current = next;
    if (!prev) {
      form.reset(next.form);
      setTags(next.tags);
      return;
    }
    const current = form.getValues();
    for (const key of profileFields)
      if (current[key] === prev.form[key] && next.form[key] !== prev.form[key])
        form.setValue(key, next.form[key]);
    setTags((old) => {
      const changed = tagGroups.filter(
        (key) =>
          sameTags(old[key], prev.tags[key]) &&
          !sameTags(next.tags[key], prev.tags[key]),
      );
      if (!changed.length) return old;
      const merged = { ...old };
      for (const key of changed) merged[key] = next.tags[key];
      return merged;
    });
  }, [query.data, form]);
  // 從其他管道匯入的檔案可能使用清單外的城市，保留它原本的座標。
  const saved = query.data;
  const cityOptions =
    saved && !cities.some((c) => c.name === saved.city)
      ? [
          ...cities,
          { name: saved.city, lat: saved.latitude, lng: saved.longitude },
        ]
      : cities;
  const save = form.handleSubmit(async (values) => {
    setError("");
    setSuccess("");
    const city = cityOptions.find((c) => c.name === values.city);
    if (!city) {
      setError("請選擇居住城市。");
      return;
    }
    try {
      await send(
        "/profile",
        { ...values, latitude: city.lat, longitude: city.lng, ...tags },
        "PUT",
      );
      // 剛送出的內容就是伺服器目前的資料；重新取得時只會多出伺服器端的整理（例如去除前後空白）。
      synced.current = { form: values, tags };
      await client.invalidateQueries({ queryKey: ["/profile"] });
      await client.invalidateQueries({ queryKey: ["/discovery"] });
      setSuccess("個人檔案已儲存，現在可以開始探索。");
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
    <>
      <Heading
        overline="THIS IS YOUR STORY"
        title="真實的你，最有魅力。"
        text="一點生活、一點喜好，讓對的人更容易認識你。"
      />
      {query.isLoading ? (
        <Loading />
      ) : (
        <div className="form-layout">
          <section className="panel">
            <form onSubmit={save}>
              <div className="panel-title">
                <UserRound size={20} />
                <h2>關於我</h2>
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
                  想遇見的關係
                  <select {...form.register("datingIntent")}>
                    {Object.entries(intentLabels)
                      .filter(([k]) => k !== "any")
                      .map(([k, v]) => (
                        <option key={k} value={k}>
                          {v}
                        </option>
                      ))}
                  </select>
                </label>
                <label>
                  職業（選填）
                  <input {...form.register("occupation")} maxLength={80} />
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
              <h3>那些讓你發光的小喜好</h3>
              <p className="muted">選擇你喜歡的事，為對話留一個起點。</p>
              {(
                [
                  { key: "interests", title: "興趣", values: interests.data },
                  { key: "hobbies", title: "休閒活動", values: hobbies.data },
                  { key: "foods", title: "喜愛的食物", values: foods.data },
                ] as const
              ).map((group) => (
                <fieldset className="tag-field" key={group.key}>
                  <legend>{group.title}</legend>
                  <div className="tags selectable">
                    {group.values?.map((tag) => (
                      <button
                        type="button"
                        key={tag}
                        aria-pressed={tags[group.key].includes(tag)}
                        onClick={() =>
                          setTags({
                            ...tags,
                            [group.key]: tags[group.key].includes(tag)
                              ? tags[group.key].filter((t) => t !== tag)
                              : [...tags[group.key], tag],
                          })
                        }
                      >
                        {tags[group.key].includes(tag) && <Check size={14} />}{" "}
                        {tag}
                      </button>
                    ))}
                  </div>
                </fieldset>
              ))}
              <ErrorText message={error || query.error?.message} />
              {success && (
                <p className="success" role="status">
                  <Check size={18} />
                  {success}
                </p>
              )}
              <button className="button" disabled={form.formState.isSubmitting}>
                {form.formState.isSubmitting ? "儲存中…" : "儲存個人檔案"}
                <Check size={17} />
              </button>
            </form>
          </section>
          <aside>
            <section className="panel photos-panel">
              <h2>你的生活切片</h2>
              <p className="muted">
                {query.data
                  ? "第一張照片會成為你的主照片。最多 6 張，每張 8 MB。"
                  : "先儲存「關於我」，就能加入照片。"}
              </p>
              <div className="photo-grid">
                {query.data?.photos.map((photo) => (
                  <div className="photo-tile" key={photo.id}>
                    <img src={photo.url} alt="我的照片" />
                    <button
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
                {(query.data?.photos.length || 0) < 6 && (
                  <label
                    className={
                      query.data ? "photo-upload" : "photo-upload disabled"
                    }
                  >
                    <Camera size={23} />
                    <span>
                      {busy
                        ? "上傳中…"
                        : query.data
                          ? "加入照片"
                          : "先儲存檔案"}
                    </span>
                    <input
                      type="file"
                      accept="image/jpeg,image/png,image/webp"
                      disabled={busy || !query.data}
                      onChange={(e) => {
                        void upload(e.target.files?.[0]);
                        e.target.value = "";
                      }}
                    />
                  </label>
                )}
              </div>
            </section>
            <section className="note-card compact">
              <ShieldCheck />
              <h3>讓真實，多一份安心。</h3>
              <p>查看你的驗證狀態。只有完成真人驗證，才會顯示驗證標記。</p>
              <Link href="/verification" className="text-link">
                查看真人驗證
                <ArrowRight size={16} />
              </Link>
            </section>
          </aside>
        </div>
      )}
    </>
  );
}
function PreferencesPage() {
  const query = useData<Preferences>("/preferences");
  const blocks =
    useData<{ blockedUserId: string; displayName: string }[]>("/blocks");
  const client = useQueryClient();
  // 年齡欄位保留使用者輸入的文字，送出時才轉數字；受控的數字欄位若存成 number，清空時會被補成 0。
  const [values, setValues] = useState<
    Omit<Preferences, "minAge" | "maxAge"> & { minAge: string; maxAge: string }
  >({
    minAge: "18",
    maxAge: "99",
    preferredGender: "any",
    maxDistanceKm: 100,
    preferredDatingIntent: "any",
  });
  const [error, setError] = useState("");
  const [saved, setSaved] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (query.data) {
      const {
        minAge,
        maxAge,
        preferredGender,
        maxDistanceKm,
        preferredDatingIntent,
      } = query.data;
      setValues({
        minAge: String(minAge),
        maxAge: String(maxAge),
        preferredGender,
        maxDistanceKm,
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
        title="你的期待，值得被理解。"
        text="探索會同時考量你與對方的偏好。"
      />
      <section className="panel narrow">
        <form
          onSubmit={async (e) => {
            e.preventDefault();
            setBusy(true);
            setError("");
            setSaved(false);
            try {
              await send(
                "/preferences",
                {
                  ...values,
                  minAge: Number(values.minAge),
                  maxAge: Number(values.maxAge),
                },
                "PUT",
              );
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
            <label>
              最小年齡
              <input
                type="number"
                min={18}
                max={99}
                required
                value={values.minAge}
                onChange={(e) =>
                  setValues({ ...values, minAge: e.target.value })
                }
              />
            </label>
            <label>
              最大年齡
              <input
                type="number"
                min={values.minAge || 18}
                max={99}
                required
                value={values.maxAge}
                onChange={(e) =>
                  setValues({ ...values, maxAge: e.target.value })
                }
              />
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
                {Object.entries(intentLabels).map(([k, v]) => (
                  <option key={k} value={k}>
                    {v}
                  </option>
                ))}
              </select>
            </label>
            <label className="span-two">
              探索距離：{values.maxDistanceKm} 公里
              <input
                type="range"
                min={1}
                max={2000}
                value={Math.min(values.maxDistanceKm, 2000)}
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
        title="多一點真實，多一份安心。"
        text="真人驗證包含身分比對與活體判斷。上傳照片本身不代表通過驗證。"
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
            if (file.size > photoLimit) {
              setError(photoTooLarge);
              return;
            }
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
              onChange={(e) => setFile(e.target.files?.[0] || null)}
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
function MatchesPage() {
  const q = useData<Match[]>("/matches");
  const client = useQueryClient();
  const [error, setError] = useState("");
  return (
    <>
      <Heading
        overline="A MUTUAL LITTLE SPARK"
        title="剛好，你們也喜歡彼此。"
        text="一句簡單的你好，可能是一段好故事的開始。"
      />
      <ErrorText message={error || q.error?.message} />
      {q.isLoading ? (
        <Loading />
      ) : !q.data?.length ? (
        <Empty
          title="你的下一個火花，還在路上"
          text="到探索看看，彼此喜歡後會在這裡相遇。"
          link="/discover"
          label="開始探索"
        />
      ) : (
        <div className="match-grid">
          {q.data.map((m) => (
            <article className="match-card" key={m.id}>
              <Portrait person={m.otherUser} large />
              <div className="match-copy">
                <h2>
                  {m.otherUser.displayName} <span>{m.otherUser.age}</span>
                </h2>
                <p>
                  <MapPin size={14} />
                  {m.otherUser.city} · {intentLabels[m.otherUser.datingIntent]}
                </p>
                <div className="tags">
                  {m.otherUser.interests.slice(0, 3).map((t) => (
                    <span key={t}>{t}</span>
                  ))}
                </div>
                <Link
                  className="button full"
                  href={`/messages/${m.conversationId}`}
                >
                  <MessageCircle size={18} />
                  說聲你好
                </Link>
                <div className="sub-actions">
                  <button
                    onClick={async () => {
                      if (!window.confirm("確定結束配對？這段對話將無法繼續。"))
                        return;
                      try {
                        await api(`/matches/${m.id}`, { method: "DELETE" });
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
                      if (!window.confirm("確定封鎖對方並結束配對？")) return;
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
            </article>
          ))}
        </div>
      )}
    </>
  );
}
function MessagesPage({ id, socket }: { id?: string; socket: Socket | null }) {
  const q = useData<Conversation[]>("/conversations");
  const selected = q.data?.find((c) => c.id === id);
  return (
    <>
      <Heading
        overline="EVERY CONVERSATION IS A BEGINNING"
        title="好好說話，慢慢認識。"
        text="真誠的好奇，是最好的開場。"
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
                你的對話 <span>{q.data.length}</span>
              </div>
              {q.data.map((c) => (
                <Link
                  href={`/messages/${c.id}`}
                  className={
                    id === c.id ? "conversation selected" : "conversation"
                  }
                  key={c.id}
                >
                  <Portrait person={c.otherUser} />
                  <div>
                    <b>{c.otherUser.displayName}</b>
                    <p>{c.lastMessage?.content || "從一句你好開始吧。"}</p>
                  </div>
                  {c.unreadCount > 0 && (
                    <span className="unread">{c.unreadCount}</span>
                  )}
                </Link>
              ))}
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
  const end = useRef<HTMLDivElement>(null);
  const draft = useRef<{ content: string; clientId: string } | null>(null);
  const typingTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastTyping = useRef(0);
  useEffect(() => {
    const saved = c.otherLastReadAt;
    if (saved) setReadAt((current) => (current > saved ? current : saved));
  }, [c.otherLastReadAt]);
  // 與伺服器分頁相同的順序（時間，再比 id），最舊的一則才能當「載入較早訊息」的游標。
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
  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!content.trim() || busy || closed) return;
    setBusy(true);
    setError("");
    if (!draft.current || draft.current.content !== content.trim())
      draft.current = {
        content: content.trim(),
        clientId: crypto.randomUUID(),
      };
    try {
      const message = await send<Message>(
        `/conversations/${c.id}/messages`,
        draft.current,
      );
      merge([message]);
      setContent("");
      draft.current = null;
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
        <Portrait person={c.otherUser} />
        <div>
          <h2>{c.otherUser.displayName}</h2>
          <small>{online ? "在線上" : "離線"}</small>
        </div>
        <span className="chat-shield">
          <ShieldCheck size={17} />
          雙向配對
        </span>
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
            <Heart size={25} />
            <h3>你們的故事，從這裡開始。</h3>
            <p>「你最近最喜歡的一間咖啡店是哪間？」</p>
          </div>
        ) : (
          messages.map((m) => (
            <div
              className={m.senderId === user.id ? "message own" : "message"}
              key={m.id}
            >
              <div>{m.content}</div>
              <small>
                {new Date(m.createdAt).toLocaleTimeString("zh-TW", {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
                {m.senderId === user.id && readAt >= m.createdAt && (
                  <>
                    <CheckCheck size={13} />
                    已讀
                  </>
                )}
              </small>
            </div>
          ))
        )}
        <div ref={end} />
      </div>
      <div className="typing-line" aria-live="polite">
        {typing
          ? "對方正在輸入…"
          : closed
            ? "這段對話已結束。"
            : "留一點真誠，給每一句話。"}
      </div>
      <ErrorText message={error || q.error?.message} />
      <form className="message-input" onSubmit={submit}>
        <input
          aria-label="訊息內容"
          placeholder="說聲你好，聊聊今天…"
          value={content}
          maxLength={2000}
          disabled={closed}
          onChange={(e) => {
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
        <button
          className="button"
          aria-label="傳送訊息"
          disabled={busy || !content.trim() || closed}
        >
          <Send size={19} />
        </button>
      </form>
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
        title="有人，正在靠近你的故事。"
        text="在這裡查看新的配對與訊息。"
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
                  <Heart size={21} />
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
