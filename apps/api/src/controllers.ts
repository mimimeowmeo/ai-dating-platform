import {
  Controller,
  Get,
  Put,
  Post,
  Delete,
  Body,
  Param,
  Query,
  Req,
  Res,
  UseGuards,
  UseInterceptors,
  UploadedFile,
  UploadedFiles,
} from "@nestjs/common";
import { FileInterceptor, FilesInterceptor } from "@nestjs/platform-express";
import type { Response } from "express";
import { z } from "zod";
import {
  AuthGuard,
  AuthRequest,
  catalogs,
  Database,
  Infrastructure,
  parse,
  config,
  fail,
} from "./core";
import { Profiles } from "./profiles";
import { Social } from "./social";
import { ReplySuggestions } from "./ai-reply";
const upload = FileInterceptor("file", {
  limits: { fileSize: 8 * 1024 * 1024, files: 1 },
});
// 即時鏡頭的影格：欄位名稱 frames，最多 4 張（正面 1 張＋最多 3 個動作），每張 8 MB 以內。
const liveUpload = FilesInterceptor("frames", 4, {
  limits: { fileSize: 8 * 1024 * 1024, files: 4 },
});
@Controller()
export class HealthController {
  constructor(
    private db: Database,
    private infra: Infrastructure,
    private profiles: Profiles,
  ) {}
  @Get("health") async health() {
    await this.db.$queryRaw`SELECT 1`;
    await this.infra.redis.ping();
    if (!(await this.infra.storage.bucketExists(config.S3_BUCKET)))
      return fail(503, "STORAGE_UNAVAILABLE", "儲存服務尚未就緒。");
    return { status: "ok", service: "dating-api" };
  }
  // 簽章本身就是授權：<img> 送不出 Authorization header，所以不掛 AuthGuard。
  @Get("media/:id") async media(
    @Param("id") id: string,
    @Query() query: { u?: string; e?: string; s?: string },
    @Res() res: Response,
  ) {
    const stream = await this.profiles.media(id, query);
    res.setHeader("Content-Type", "image/jpeg");
    res.setHeader("Cache-Control", "private, max-age=300");
    stream.on("error", () => res.destroy());
    stream.pipe(res);
  }
}
@Controller()
@UseGuards(AuthGuard)
export class ProductController {
  constructor(
    private profiles: Profiles,
    private social: Social,
    private replies: ReplySuggestions,
  ) {}
  @Get("profile") profile(@Req() r: AuthRequest) {
    return this.profiles.mine(r.userId);
  }
  @Put("profile") saveProfile(@Req() r: AuthRequest, @Body() b: unknown) {
    return this.profiles.save(r.userId, b);
  }
  @Get("profile/:id") publicProfile(
    @Req() r: AuthRequest,
    @Param("id") id: string,
  ) {
    return this.profiles.public(r.userId, id);
  }
  @Post("profile/photos") @UseInterceptors(upload) uploadPhoto(
    @Req() r: AuthRequest,
    @UploadedFile() file: Express.Multer.File,
  ) {
    return this.profiles.upload(r.userId, file);
  }
  @Delete("profile/photos/:id") removePhoto(
    @Req() r: AuthRequest,
    @Param("id") id: string,
  ) {
    return this.profiles.removePhoto(r.userId, id);
  }
  @Get("preferences") prefs(@Req() r: AuthRequest) {
    return this.profiles.preferences(r.userId);
  }
  @Put("preferences") savePrefs(@Req() r: AuthRequest, @Body() b: unknown) {
    return this.profiles.savePreferences(r.userId, b);
  }
  @Get("traits") traits() {
    return this.profiles.traitCatalog();
  }
  @Get("interests") interests() {
    return catalogs.interests;
  }
  @Get("hobbies") hobbies() {
    return catalogs.hobbies;
  }
  @Get("foods") foods() {
    return catalogs.foods;
  }
  @Put("me/interests") setInterests(@Req() r: AuthRequest, @Body() b: unknown) {
    return this.profiles.setTags(r.userId, "interests", b);
  }
  @Put("me/hobbies") setHobbies(@Req() r: AuthRequest, @Body() b: unknown) {
    return this.profiles.setTags(r.userId, "hobbies", b);
  }
  @Put("me/foods") setFoods(@Req() r: AuthRequest, @Body() b: unknown) {
    return this.profiles.setTags(r.userId, "foods", b);
  }
  /**
   * GET /verification/status：查詢自己的真人驗證狀態。
   * 回傳最近一筆驗證紀錄，以及 canVerify（已上傳第一張大頭貼才是 true）；
   * 前端依 canVerify 決定要不要顯示真人驗證區塊。
   */
  @Get("verification/status") verification(@Req() r: AuthRequest) {
    // r.userId 是登入 token 解出來的使用者編號（這個 controller 掛了 AuthGuard，未登入進不來）。
    return this.profiles.verification(r.userId);
  }
  /**
   * POST /onboarding/selfie：上傳自拍做真人驗證（multipart 表單，欄位名稱 file）。
   * 還沒上傳大頭貼會回 409 AVATAR_REQUIRED；檔案超過 8 MB 由 upload 攔截器回 413。
   */
  @Post("onboarding/selfie") @UseInterceptors(upload) selfie(
    // 已登入的請求，帶有使用者編號。
    @Req() r: AuthRequest,
    // upload 攔截器（multer）解析出來的檔案；沒附檔案時是 undefined，由 verifySelfie 回 400。
    @UploadedFile() file: Express.Multer.File,
  ) {
    // 交給 Profiles.verifySelfie：檢查大頭貼、整理自拍、呼叫 AI 服務、寫入結果。
    return this.profiles.verifySelfie(r.userId, file);
  }
  /**
   * POST /verification/challenge：開始即時鏡頭驗證，領一份隨機的動作挑戰（120 秒內有效、只能用一次）。
   * 還沒上傳大頭貼會回 409 AVATAR_REQUIRED。
   */
  @Post("verification/challenge") challenge(@Req() r: AuthRequest) {
    // 交給 Profiles.createChallenge：檢查大頭貼、抽動作、存進 Redis。
    return this.profiles.createChallenge(r.userId);
  }
  /**
   * POST /onboarding/live：送出即時鏡頭拍下的影格（multipart 表單）。
   * 欄位 challengeId 是挑戰編號；欄位 frames 依序放「正面影格」與每個動作各一張。
   * liveUpload 攔截器最多收 4 張、每張 8 MB 以內，超過回 413／400。
   */
  @Post("onboarding/live") @UseInterceptors(liveUpload) live(
    // 已登入的請求，帶有使用者編號。
    @Req() r: AuthRequest,
    // liveUpload 攔截器（multer）解析出來的影格陣列；沒附檔案時是 undefined。
    @UploadedFiles() files: Express.Multer.File[],
    // 表單裡的文字欄位（challengeId）；由 verifyLive 檢查格式。
    @Body() body: unknown,
  ) {
    // 交給 Profiles.verifyLive：核對挑戰、整理影格、呼叫 AI 服務、寫入結果。
    return this.profiles.verifyLive(r.userId, files, body);
  }
  /**
   * POST /verification/retry：按「重新驗證」。
   * 不寫資料庫，只回傳接下來要做什麼（先上傳大頭貼，或用鏡頭即時驗證）。
   */
  @Post("verification/retry") retry(@Req() r: AuthRequest) {
    // 交給 Profiles.retry 判斷目前能不能驗證。
    return this.profiles.retry(r.userId);
  }
  @Get("discovery") discovery(@Req() r: AuthRequest) {
    return this.social.discovery(r.userId);
  }
  // 測試用搜尋：GET /discovery/search?q=名稱或 email。
  @Get("discovery/search") search(
    @Req() r: AuthRequest,
    @Query("q") q: unknown,
  ) {
    return this.social.search(r.userId, q);
  }
  // 測試用：從搜尋列按喜歡／略過，規則見 Social.interact 的 test 模式。
  @Post("discovery/search/interactions") searchInteraction(
    @Req() r: AuthRequest,
    @Body() b: unknown,
  ) {
    return this.social.interact(r.userId, b, true);
  }
  @Post("interactions") interaction(@Req() r: AuthRequest, @Body() b: unknown) {
    return this.social.interact(r.userId, b);
  }
  // 我按過「喜歡」的人，含「還在等對方回應」與「已配對」兩種狀態。
  @Get("likes") likes(@Req() r: AuthRequest) {
    return this.social.likesSent(r.userId);
  }
  @Get("matches") matches(@Req() r: AuthRequest) {
    return this.social.matches(r.userId);
  }
  @Get("matches/:id") match(@Req() r: AuthRequest, @Param("id") id: string) {
    return this.social.matches(r.userId, id);
  }
  @Delete("matches/:id") unmatch(
    @Req() r: AuthRequest,
    @Param("id") id: string,
  ) {
    return this.social.unmatch(r.userId, id);
  }
  @Get("blocks") blocks(@Req() r: AuthRequest) {
    return this.social.blocks(r.userId);
  }
  @Post("blocks") block(@Req() r: AuthRequest, @Body() b: unknown) {
    return this.social.block(
      r.userId,
      parse(z.object({ blockedUserId: z.string().uuid() }).strict(), b)
        .blockedUserId,
    );
  }
  @Delete("blocks/:id") unblock(
    @Req() r: AuthRequest,
    @Param("id") id: string,
  ) {
    return this.social.unblock(r.userId, id);
  }
  @Get("conversations") conversations(@Req() r: AuthRequest) {
    return this.social.conversations(r.userId);
  }
  @Post("conversations") createConversation(
    @Req() r: AuthRequest,
    @Body() b: unknown,
  ) {
    return this.social.createConversation(r.userId, b);
  }
  @Get("conversations/:id") conversation(
    @Req() r: AuthRequest,
    @Param("id") id: string,
  ) {
    return this.social.conversation(r.userId, id);
  }
  @Get("conversations/:id/messages") messages(
    @Req() r: AuthRequest,
    @Param("id") id: string,
    @Query("before") before?: string,
    @Query("beforeId") beforeId?: string,
  ) {
    return this.social.messages(r.userId, id, before, beforeId);
  }
  @Post("conversations/:id/messages") send(
    @Req() r: AuthRequest,
    @Param("id") id: string,
    @Body() b: unknown,
  ) {
    return this.social.send(r.userId, id, b);
  }
  @Post("conversations/:id/read") read(
    @Req() r: AuthRequest,
    @Param("id") id: string,
  ) {
    return this.social.read(r.userId, id);
  }
  // 按一次「AI 推薦回覆」：回 3～5 則建議，第 1 則給前端用打字動畫填進輸入框。
  // 每一則都帶 id，送訊息時當作 suggestionId 回傳，後端才能標記訊息來源。
  @Post("conversations/:id/reply-suggestions") replySuggestions(
    @Req() r: AuthRequest,
    @Param("id") id: string,
  ) {
    return this.replies.suggest(r.userId, id);
  }
  @Get("notifications") notifications(@Req() r: AuthRequest) {
    return this.social.notifications(r.userId);
  }
  @Post("notifications/:id/read") readNotification(
    @Req() r: AuthRequest,
    @Param("id") id: string,
  ) {
    return this.social.readNotification(r.userId, id);
  }
}
