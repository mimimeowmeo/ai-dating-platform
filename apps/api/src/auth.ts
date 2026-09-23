import {
  Injectable,
  Controller,
  Post,
  Get,
  Body,
  Req,
  Res,
  UseGuards,
} from "@nestjs/common";
import { hash, compare } from "bcryptjs";
import { randomBytes, createHash } from "node:crypto";
import { sign } from "jsonwebtoken";
import { z } from "zod";
import type { Request, Response } from "express";
import {
  allowedWebOrigins,
  Database,
  Infrastructure,
  config,
  fail,
  parse,
  AuthGuard,
  AuthRequest,
} from "./core";
import { ipBucket } from "./ip-bucket";
const credentials = z
  .object({
    email: z
      .string()
      .trim()
      .email()
      .max(254)
      .transform((v) => v.toLowerCase()),
    password: z
      .string()
      .min(12, "密碼至少 12 個字元")
      .max(72, "密碼最多 72 個字元")
      .refine(
        (v) => Buffer.byteLength(v) <= 72,
        "密碼 UTF-8 長度不得超過 72 位元組",
      ),
  })
  .strict();
const digest = (v: string) => createHash("sha256").update(v).digest("hex");
const publicUser = (u: any) => ({
  id: u.id,
  email: u.email,
  isVerified: u.isVerified,
});
const REFRESH_OVERLAP_MS = 30_000;
const accessToken = (userId: string, sessionId: string) =>
  sign({ sid: sessionId }, config.JWT_SECRET, {
    subject: userId,
    expiresIn: "15m",
    issuer: "dating-api",
    audience: "dating-web",
  });
@Injectable()
export class AuthService {
  constructor(
    private db: Database,
    private infra: Infrastructure,
  ) {}
  checkOrigin(req: Request) {
    if (req.headers.origin && !allowedWebOrigins.includes(req.headers.origin))
      fail(403, "ORIGIN_DENIED", "請從本站登入。");
  }
  async createSession(user: any) {
    const token = randomBytes(48).toString("hex");
    const session = await this.db.session.create({
      data: {
        userId: user.id,
        tokenHash: digest(token),
        expiresAt: new Date(Date.now() + 30 * 864e5),
      },
    });
    return {
      token,
      accessToken: accessToken(user.id, session.id),
      user: publicUser(user),
    };
  }
  async register(body: unknown, ip: string) {
    await this.infra.limit(`auth:${ip}`, 20, 300);
    const dto = parse(credentials, body);
    const user = await this.db.user.create({
      data: {
        email: dto.email,
        passwordHash: await hash(dto.password, 12),
        preference: { create: {} },
      },
    });
    return this.createSession(user);
  }
  async login(body: unknown, ip: string) {
    await this.infra.limit(`auth:${ip}`, 20, 300);
    const dto = parse(credentials, body);
    const user = await this.db.user.findUnique({ where: { email: dto.email } });
    const valid = await compare(
      dto.password,
      user?.passwordHash ||
        "$2b$12$N9qo8uLOickgx2ZMRZoMyeIjZAgcfl7p92ldGxad68LJZdL17lhWy",
    );
    if (!user || !valid)
      return fail(401, "INVALID_CREDENTIALS", "電子郵件或密碼不正確。");
    return this.createSession(user);
  }
  async refresh(token: string) {
    if (!token || token.length > 200)
      return fail(401, "UNAUTHORIZED", "請重新登入。");
    const hash = digest(token);
    return this.db.$transaction(async (tx) => {
      const found = await tx.session.findFirst({
        where: { OR: [{ tokenHash: hash }, { previousTokenHash: hash }] },
        select: { id: true },
      });
      if (!found) return fail(401, "UNAUTHORIZED", "登入已失效。");
      await tx.$executeRaw`SELECT pg_advisory_xact_lock(hashtext(${`session:${found.id}`}))`;
      const session = await tx.session.findUniqueOrThrow({
        where: { id: found.id },
        include: { user: true },
      });
      if (session.revokedAt || session.expiresAt < new Date())
        return fail(401, "UNAUTHORIZED", "登入已失效。");
      const now = new Date();
      const recentlyRotated =
        !!session.rotatedAt &&
        now.getTime() - session.rotatedAt.getTime() < REFRESH_OVERLAP_MS;
      const isCurrent = session.tokenHash === hash;
      if (
        !isCurrent &&
        !(session.previousTokenHash === hash && recentlyRotated)
      )
        return fail(401, "UNAUTHORIZED", "登入已更新，請重新登入。");
      let next: string | undefined;
      if (isCurrent && !recentlyRotated) {
        next = randomBytes(48).toString("hex");
        await tx.session.update({
          where: { id: session.id },
          data: {
            previousTokenHash: hash,
            tokenHash: digest(next),
            rotatedAt: now,
            expiresAt: new Date(now.getTime() + 30 * 864e5),
          },
        });
      }
      return {
        token: next,
        accessToken: accessToken(session.userId, session.id),
        user: publicUser(session.user),
      };
    });
  }
  async logout(token: string) {
    if (token) {
      const hash = digest(token);
      await this.db.session.updateMany({
        where: {
          OR: [{ tokenHash: hash }, { previousTokenHash: hash }],
          revokedAt: null,
        },
        data: { revokedAt: new Date() },
      });
    }
    return { ok: true };
  }
  async me(id: string) {
    const user = await this.db.user.findUniqueOrThrow({ where: { id } });
    return publicUser(user);
  }
}
@Controller("auth")
export class AuthController {
  constructor(
    private auth: AuthService,
    private infra: Infrastructure,
  ) {}
  cookie(res: Response, data: any) {
    if (data.token)
      res.cookie("dating_refresh", data.token, {
        httpOnly: true,
        secure: config.COOKIE_SECURE === "true",
        sameSite: "lax",
        path: "/api/v1/auth",
        maxAge: 30 * 864e5,
      });
    return { accessToken: data.accessToken, user: data.user };
  }
  @Post("register") async register(
    @Body() body: unknown,
    @Req() req: Request,
    @Res({ passthrough: true }) res: Response,
  ) {
    this.auth.checkOrigin(req);
    return this.cookie(res, await this.auth.register(body, ipBucket(req.ip)));
  }
  @Post("login") async login(
    @Body() body: unknown,
    @Req() req: Request,
    @Res({ passthrough: true }) res: Response,
  ) {
    this.auth.checkOrigin(req);
    return this.cookie(res, await this.auth.login(body, ipBucket(req.ip)));
  }
  @Post("refresh") async refresh(
    @Req() req: Request,
    @Res({ passthrough: true }) res: Response,
  ) {
    this.auth.checkOrigin(req);
    await this.infra.limit(`refresh:${ipBucket(req.ip)}`, 60, 60);
    return this.cookie(
      res,
      await this.auth.refresh(req.cookies?.dating_refresh),
    );
  }
  @Post("logout") async logout(
    @Req() req: Request,
    @Res({ passthrough: true }) res: Response,
  ) {
    this.auth.checkOrigin(req);
    const out = await this.auth.logout(req.cookies?.dating_refresh);
    res.clearCookie("dating_refresh", {
      httpOnly: true,
      secure: config.COOKIE_SECURE === "true",
      sameSite: "lax",
      path: "/api/v1/auth",
    });
    return out;
  }
  @Get("me") @UseGuards(AuthGuard) me(@Req() req: AuthRequest) {
    return this.auth.me(req.userId);
  }
}
