import {
  WebSocketGateway,
  WebSocketServer,
  OnGatewayInit,
  OnGatewayConnection,
  OnGatewayDisconnect,
  SubscribeMessage,
  ConnectedSocket,
  MessageBody,
} from "@nestjs/websockets";
import { Server, Socket } from "socket.io";
import { HttpException } from "@nestjs/common";
import { z } from "zod";
import { Social } from "./social";
import { Identity, Infrastructure, allowedWebOrigins, parse } from "./core";
const OFFLINE_GRACE_MS = 3000;
@WebSocketGateway({
  cors: { origin: allowedWebOrigins, credentials: true },
  maxHttpBufferSize: 32_768,
})
export class Realtime
  implements OnGatewayInit, OnGatewayConnection, OnGatewayDisconnect
{
  @WebSocketServer() server!: Server;
  private online = new Map<string, Set<string>>();
  constructor(
    private identity: Identity,
    private social: Social,
    private infra: Infrastructure,
  ) {}
  afterInit(server: Server) {
    server.use(async (socket, next) => {
      try {
        Object.assign(
          socket.data,
          await this.identity.authenticate(socket.handshake.auth.token || ""),
        );
        next();
      } catch {
        next(new Error("請重新登入。"));
      }
    });
    this.social.publish = (id, event, data) => {
      void (async () => {
        const clients = await server.in(`user:${id}`).fetchSockets();
        if (!clients.length) return;
        const allowed =
          event !== "message:new" ||
          (await this.social
            .access(id, (data as { conversationId: string }).conversationId)
            .then(
              () => true,
              () => false,
            ));
        for (const client of clients) {
          try {
            await this.identity.authenticate(client.handshake.auth.token || "");
          } catch {
            client.disconnect(true);
            continue;
          }
          if (allowed) client.emit(event, data);
        }
      })().catch(() => {});
    };
    this.social.revoke = (id) => {
      server
        .to(`conversation:${id}`)
        .emit("conversation:closed", { conversationId: id });
      server.in(`conversation:${id}`).socketsLeave(`conversation:${id}`);
    };
  }
  /**
   * 上線／離線時通知每一位聊天對象。
   * 發到對方的 user 房間（而不是聊天室房間）：對方只要登入就收得到，
   * 不必先打開那個聊天室，對話列表才能顯示每個人的上線狀態。
   */
  private announcePresence(userId: string, online: boolean) {
    void this.social
      .activePartners(userId)
      .then((partners) => {
        for (const { conversationId, otherUserId } of partners)
          this.server
            .to(`user:${otherUserId}`)
            .emit("presence", { conversationId, userId, online });
      })
      .catch(() => {});
  }
  async handleConnection(socket: Socket) {
    const userId: string = socket.data.userId;
    socket.join(`user:${userId}`);
    const connections = this.online.get(userId) || new Set<string>();
    const cameOnline = connections.size === 0;
    connections.add(socket.id);
    this.online.set(userId, connections);
    if (cameOnline) this.announcePresence(userId, true);
  }
  handleDisconnect(socket: Socket) {
    const userId: string = socket.data.userId;
    const connections = this.online.get(userId);
    connections?.delete(socket.id);
    if (connections && !connections.size) {
      this.online.delete(userId);
      setTimeout(() => {
        if (!this.online.has(userId)) this.announcePresence(userId, false);
      }, OFFLINE_GRACE_MS).unref();
    }
  }
  async run(socket: Socket, fn: (id: string) => Promise<unknown>) {
    try {
      const { userId } = await this.identity.authenticate(
        socket.handshake.auth.token || "",
      );
      await this.infra.limit(`socket:${userId}`, 180);
      return { ok: true, ...((await fn(userId)) as object) };
    } catch (e) {
      const error =
        e instanceof HttpException
          ? e.getResponse()
          : { message: "操作失敗，請稍後再試。" };
      return { ok: false, error };
    }
  }
  @SubscribeMessage("conversation:join") join(
    @ConnectedSocket() socket: Socket,
    @MessageBody() body: unknown,
  ) {
    return this.run(socket, async (id) => {
      const dto = parse(
        z.object({ conversationId: z.string().uuid() }).strict(),
        body,
      );
      const c = await this.social.access(id, dto.conversationId);
      socket.join(`conversation:${c.id}`);
      socket.emit("presence", {
        conversationId: c.id,
        userId: c.otherUserId,
        online: this.online.has(c.otherUserId),
      });
      socket
        .to(`conversation:${c.id}`)
        .emit("presence", { conversationId: c.id, userId: id, online: true });
      return {};
    });
  }
  /**
   * 對話列表一次問出「我的聊天對象現在誰在線上」。
   * presence 事件只有狀態變化時才會發，剛進頁面時需要這一份快照。
   */
  @SubscribeMessage("presence:list") presenceList(
    @ConnectedSocket() socket: Socket,
  ) {
    return this.run(socket, async (id) => {
      const partners = await this.social.activePartners(id);
      return {
        online: [
          ...new Set(
            partners
              .map((partner) => partner.otherUserId)
              .filter((userId) => this.online.has(userId)),
          ),
        ],
      };
    });
  }
  @SubscribeMessage("message:send") send(
    @ConnectedSocket() socket: Socket,
    @MessageBody() body: unknown,
  ) {
    return this.run(socket, async (id) => {
      const dto = parse(
        z
          .object({
            conversationId: z.string().uuid(),
            content: z.string(),
            clientId: z.string().uuid(),
            // 與 HTTP 的送訊息同一組欄位：帶著推薦 id 才能標記訊息來源（規格 5.6）。
            suggestionId: z.string().uuid().optional(),
          })
          .strict(),
        body,
      );
      return {
        message: await this.social.send(id, dto.conversationId, {
          content: dto.content,
          clientId: dto.clientId,
          ...(dto.suggestionId ? { suggestionId: dto.suggestionId } : {}),
        }),
      };
    });
  }
  @SubscribeMessage("typing") typing(
    @ConnectedSocket() socket: Socket,
    @MessageBody() body: unknown,
  ) {
    return this.run(socket, async (id) => {
      const dto = parse(
        z
          .object({ conversationId: z.string().uuid(), isTyping: z.boolean() })
          .strict(),
        body,
      );
      await this.social.access(id, dto.conversationId);
      socket
        .to(`conversation:${dto.conversationId}`)
        .emit("typing", { ...dto, userId: id });
      return {};
    });
  }
  @SubscribeMessage("conversation:read") read(
    @ConnectedSocket() socket: Socket,
    @MessageBody() body: unknown,
  ) {
    return this.run(socket, async (id) => {
      const dto = parse(
        z.object({ conversationId: z.string().uuid() }).strict(),
        body,
      );
      await this.social.read(id, dto.conversationId);
      return {};
    });
  }
}
