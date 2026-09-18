// 選項清單（對應後端 traits 資料表）。畫面會向 GET /traits 取得最新內容，
// 這份快照只是後端還沒回應前的預設值，共 68 筆。
export type TraitRow = { category: string; code: string; label: string };
export const traitSnapshot: TraitRow[] = [
  { category: "dating_goal", code: "serious_relationship", label: "認真交往" },
  { category: "dating_goal", code: "friends_first", label: "先做朋友" },
  { category: "dating_goal", code: "chat_only", label: "只想聊天" },
  { category: "dating_goal", code: "dining_partner", label: "找飯友" },
  { category: "dating_goal", code: "marriage_minded", label: "以結婚為前提" },
  { category: "interest", code: "hiking", label: "登山" },
  { category: "interest", code: "camping", label: "露營" },
  { category: "interest", code: "surfing", label: "衝浪" },
  { category: "interest", code: "fitness", label: "健身" },
  { category: "interest", code: "running", label: "跑步" },
  { category: "interest", code: "skateboarding", label: "滑板" },
  { category: "interest", code: "badminton", label: "羽球" },
  { category: "interest", code: "basketball", label: "籃球" },
  { category: "interest", code: "cycling", label: "騎車" },
  { category: "interest", code: "swimming", label: "游泳" },
  { category: "interest", code: "yoga", label: "瑜伽" },
  { category: "interest", code: "skiing", label: "滑雪" },
  { category: "interest", code: "tv_series", label: "追劇" },
  { category: "interest", code: "movies", label: "電影" },
  { category: "interest", code: "anime", label: "動漫" },
  { category: "interest", code: "gaming", label: "電玩" },
  { category: "interest", code: "board_games", label: "桌遊" },
  { category: "interest", code: "reading", label: "閱讀" },
  { category: "interest", code: "cooking", label: "料理" },
  { category: "interest", code: "karaoke", label: "KTV" },
  { category: "interest", code: "photography", label: "攝影" },
  { category: "interest", code: "exhibitions", label: "看展" },
  { category: "interest", code: "live_music", label: "live 音樂" },
  { category: "interest", code: "singing", label: "唱歌" },
  { category: "interest", code: "playing_instruments", label: "樂器" },
  { category: "interest", code: "writing", label: "寫作" },
  { category: "interest", code: "drawing", label: "繪畫" },
  { category: "interest", code: "dancing", label: "跳舞" },
  { category: "interest", code: "exploring_shops", label: "逛店" },
  { category: "interest", code: "food", label: "美食" },
  { category: "interest", code: "wine_tasting", label: "品酒" },
  { category: "interest", code: "cat_person", label: "貓派" },
  { category: "interest", code: "dog_person", label: "狗派" },
  { category: "interest", code: "travel", label: "旅行" },
  { category: "interest", code: "coffee", label: "咖啡" },
  { category: "interest", code: "desserts", label: "甜點" },
  { category: "interest", code: "fashion", label: "時尚" },
  { category: "personality", code: "humorous", label: "幽默" },
  { category: "personality", code: "slow_to_warm_up", label: "慢熱" },
  { category: "personality", code: "talkative", label: "健談" },
  { category: "personality", code: "quiet", label: "文靜" },
  { category: "personality", code: "rational", label: "理性" },
  { category: "personality", code: "emotional", label: "感性" },
  { category: "personality", code: "optimistic", label: "樂觀" },
  { category: "personality", code: "independent", label: "獨立" },
  { category: "personality", code: "direct", label: "直接" },
  { category: "personality", code: "romantic", label: "浪漫" },
  { category: "personality", code: "action_oriented", label: "行動派" },
  { category: "personality", code: "homebody", label: "居家" },
  { category: "diet", code: "likes_seafood", label: "愛海鮮" },
  { category: "diet", code: "likes_japanese_food", label: "愛日式" },
  { category: "diet", code: "likes_hotpot", label: "愛火鍋" },
  { category: "diet", code: "likes_yakiniku", label: "愛燒肉" },
  { category: "diet", code: "vegetarian", label: "素食" },
  { category: "lifestyle", code: "nine_to_five", label: "朝九晚五" },
  { category: "lifestyle", code: "shift_work", label: "輪班" },
  { category: "lifestyle", code: "two_days_off_weekly", label: "週休二日" },
  { category: "lifestyle", code: "work_from_home", label: "遠端工作" },
  { category: "value", code: "likes_sharing_daily_life", label: "愛分享日常" },
  { category: "value", code: "values_companionship", label: "重視陪伴" },
  { category: "value", code: "needs_personal_space", label: "需要個人空間" },
  { category: "value", code: "values_communication", label: "重視溝通" },
  { category: "value", code: "values_trust", label: "重視信任" },
];
// 小標＝category 第一個底線前的字翻成中文（dating_goal → dating → 交友目標）。
const categoryTitles: Record<string, string> = {
  personality: "個性",
  diet: "飲食",
  value: "價值觀",
  lifestyle: "生活型態",
  interest: "興趣",
  dating: "交友目標",
};
export const categoryTitle = (category: string) =>
  categoryTitles[category.split("_")[0]] ?? category;
// 「我的小熱愛」顯示的類別順序；資料庫新增的類別接在後面（dating_goal 另外處理）。
const categoryOrder = ["personality", "diet", "value", "lifestyle", "interest"];
export const traitCategories = (rows: TraitRow[]) => {
  const found = [...new Set(rows.map((row) => row.category))].filter(
    (category) => category !== "dating_goal",
  );
  return [
    ...categoryOrder.filter((category) => found.includes(category)),
    ...found.filter((category) => !categoryOrder.includes(category)),
  ];
};
// 把已選的代碼依類別分組，順序與編輯頁一致。
export const groupTraits = (rows: TraitRow[], codes: string[]) => {
  const categoryOf = new Map(rows.map((row) => [row.code, row.category]));
  const groups = new Map<string, string[]>();
  for (const code of codes) {
    const category = categoryOf.get(code) ?? "";
    groups.set(category, [...(groups.get(category) ?? []), code]);
  }
  const order = traitCategories(rows);
  return [
    ...order.filter((category) => groups.has(category)),
    ...[...groups.keys()].filter((category) => !order.includes(category)),
  ].map((category) => ({ category, codes: groups.get(category) ?? [] }));
};
export const datingGoalLimit = 2;
export const traitsOf = (rows: TraitRow[], category: string) =>
  rows.filter((row) => row.category === category);
let labels = new Map(traitSnapshot.map((row) => [row.code, row.label]));
export const traitLabel = (code: string) => labels.get(code) ?? code;
// 取得後端清單後更新對照表，讓已選的代碼也能顯示正確名稱。
export const rememberTraitLabels = (rows: TraitRow[]) => {
  labels = new Map(rows.map((row) => [row.code, row.label]));
};
