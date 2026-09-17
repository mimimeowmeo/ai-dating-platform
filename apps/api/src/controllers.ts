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
} from "@nestjs/common";
import { FileInterceptor } from "@nestjs/platform-express";
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
const upload = FileInterceptor("file", {
  limits: { fileSize: 8 * 1024 * 1024, files: 1 },
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
  @Get("media/:id") async media(@Param("id") id: string, @Res() res: Response) {
    const stream = await this.profiles.media(id);
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
  @Get("verification/status") verification(@Req() r: AuthRequest) {
    return this.profiles.verification(r.userId);
  }
  @Post("onboarding/selfie") @UseInterceptors(upload) selfie(
    @Req() r: AuthRequest,
    @UploadedFile() file: Express.Multer.File,
  ) {
    return this.profiles.verifySelfie(r.userId, file);
  }
  @Post("verification/retry") retry(@Req() r: AuthRequest) {
    return this.profiles.retry(r.userId);
  }
  @Get("discovery") discovery(@Req() r: AuthRequest) {
    return this.social.discovery(r.userId);
  }
  @Post("interactions") interaction(@Req() r: AuthRequest, @Body() b: unknown) {
    return this.social.interact(r.userId, b);
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
