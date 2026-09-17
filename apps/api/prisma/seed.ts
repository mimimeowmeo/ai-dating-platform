import { PrismaClient } from "@prisma/client";
import { hash } from "bcryptjs";
const db = new PrismaClient();
async function main() {
  const password = process.env.DEMO_PASSWORD;
  if (
    process.env.NODE_ENV === "production" ||
    process.env.SEED_DEMO !== "true" ||
    !password ||
    password.length < 12
  )
    throw new Error(
      "請在本機設定 SEED_DEMO=true 與至少 12 字元的 DEMO_PASSWORD，才能建立示範資料。",
    );
  const passwordHash = await hash(password, 12);
  const examples = [
    {
      email: "demo-lin@example.test",
      name: "示範・小林",
      gender: "woman",
      interests: ["咖啡", "藝術"],
      hobbies: ["攝影"],
      foods: ["日式料理"],
    },
    {
      email: "demo-yu@example.test",
      name: "示範・阿宇",
      gender: "man",
      interests: ["旅行", "音樂"],
      hobbies: ["登山"],
      foods: ["台灣小吃"],
    },
    {
      email: "demo-an@example.test",
      name: "示範・安安",
      gender: "nonbinary",
      interests: ["閱讀", "電影"],
      hobbies: ["烹飪"],
      foods: ["蔬食"],
    },
  ];
  let count = 0;
  for (const e of examples) {
    if (await db.user.findUnique({ where: { email: e.email } })) continue;
    await db.user.create({
      data: {
        email: e.email,
        passwordHash,
        preference: { create: { maxDistanceKm: 2000 } },
        profile: {
          create: {
            displayName: e.name,
            birthDate: new Date("1997-03-15"),
            gender: e.gender,
            bio: "這是本機示範帳號，非真實交友對象。",
            city: "台北市",
            latitude: 25.033,
            longitude: 121.5654,
            datingIntent: "serious",
            interests: e.interests,
            hobbies: e.hobbies,
            foods: e.foods,
          },
        },
      },
    });
    count++;
  }
  console.log(
    `新增 ${count} 個明確標記的本機示範帳號；既有帳號保持原樣。密碼使用你提供的 DEMO_PASSWORD。`,
  );
}
main()
  .catch((e) => {
    console.error(e.message);
    process.exitCode = 1;
  })
  .finally(() => db.$disconnect());
