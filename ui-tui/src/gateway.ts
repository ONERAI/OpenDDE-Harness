// Typed calls against the Python gateway, plus routing for the five top-level
// notifications it can push (`approval.request`, `approval.closed`,
// `clarify.request`, `confirm.request`, `login.step`).
//
// Wire results come from the shared OpenRPC schema.

import type {
  ApprovalRespondResult,
  ClarifyRespondResult,
  CommandsCatalogResult,
  ConfirmRespondResult,
  ConfigGetResult,
  ConfigSetResult,
  CompletePathResult,
  ConfigSetParams,
  ModelDeclareProviderParams,
  ModelDeclareProviderResult,
  ModelLoginAnswerParams,
  ModelLoginAnswerResult,
  ModelLoginCancelParams,
  ModelLoginCancelResult,
  ModelLoginParams,
  ModelLoginResult,
  ModelLogoutParams,
  ModelLogoutResult,
  ModelOptionsParams,
  ModelOptionsResult,
  ModelOverlayParams,
  ModelOverlayResult,
  ModelSaveKeyParams,
  ModelSaveKeyResult,
  ModelScopeParams,
  ModelScopeResult,
  ReloadMcpResult,
  ServerNotification,
  SessionBranchResult,
  SessionCloseResult,
  SessionCreateResult,
  SessionDeleteResult,
  SessionExportResult,
  SessionInfoResult,
  SessionInstructionsParams,
  SessionInstructionsResult,
  SessionListResult,
  SessionMostRecentResult,
  SessionResumeResult,
  SessionStatusResult,
  SessionTitleResult,
  SessionUndoResult,
  SetupStatusResult,
  SlashExecResult,
  SystemHelloResult,
  TerminalResizeResult,
  TurnCancelResult,
  TurnEvent,
  TurnSendResult
} from './rpc/index.js'

const CLIENT_VERSION = '0.0.1'
const CLIENT_CAPABILITIES = ['cli-dispatch', 'cli', 'config']

/** The slice of `RpcClient` the gateway needs; lets tests drive it with a fake. */
export interface RpcTransport {
  rpc<R = unknown, P = unknown>(method: string, params: P): Promise<R>
  subscribe<E = unknown, P = unknown>(
    method: string,
    params: P,
    handler: (event: E) => void,
    opts?: { unsubscribeMethod?: string }
  ): Promise<{ subscription_id: string; unsubscribe: () => Promise<void> }>
}

/** Subscription events are routed separately by RpcClient. */
export type GatewayNotification = Exclude<ServerNotification, { method: 'event' }>

/** One step of a sign-in as it is pushed: pi's own step, plus whose login it is. */
export type LoginStepPush = Extract<ServerNotification, { method: 'login.step' }>['params']

export class Gateway {
  /** Server-pushed notifications that are not subscription events. */
  onNotification?: (method: GatewayNotification['method'], params: GatewayNotification['params']) => void

  constructor(private readonly transport: RpcTransport) {}

  hello(): Promise<SystemHelloResult> {
    return this.transport.rpc<SystemHelloResult>('system.hello', {
      client_version: CLIENT_VERSION,
      client_capabilities: CLIENT_CAPABILITIES
    })
  }

  commandsCatalog(): Promise<CommandsCatalogResult> {
    return this.transport.rpc<CommandsCatalogResult>('commands.catalog', {})
  }

  /** Every whitelisted config key (the handler returns all of them when
   *  `keys` is absent). */
  configGet(): Promise<ConfigGetResult> {
    return this.transport.rpc<ConfigGetResult>('config.get', {})
  }

  setupStatus(): Promise<SetupStatusResult> {
    return this.transport.rpc<SetupStatusResult>('setup.status', {})
  }

  sessionCreate(): Promise<SessionCreateResult> {
    return this.transport.rpc<SessionCreateResult>('session.create', {})
  }

  sessionResume(sessionId: string): Promise<SessionResumeResult> {
    return this.transport.rpc<SessionResumeResult>('session.resume', { session_id: sessionId })
  }

  sessionMostRecent(): Promise<SessionMostRecentResult> {
    return this.transport.rpc<SessionMostRecentResult>('session.most_recent', {})
  }

  /** The init bundle again, for a session already open. A live model switch
   *  changes several of its facts and re-sends none of them. */
  sessionInfo(sessionId: null | string): Promise<SessionInfoResult> {
    return this.transport.rpc<SessionInfoResult>('session.info', sessionId ? { session_id: sessionId } : {})
  }

  turnSubscribe(sessionKey: string, handler: (event: TurnEvent) => void) {
    return this.transport.subscribe<TurnEvent>('turn.subscribe', { session_key: sessionKey }, handler, {
      unsubscribeMethod: 'turn.unsubscribe'
    })
  }

  turnSend(sessionKey: string, content: string): Promise<TurnSendResult> {
    return this.transport.rpc<TurnSendResult>('turn.send', { session_key: sessionKey, content })
  }

  turnCancel(sessionKey: string): Promise<TurnCancelResult> {
    return this.transport.rpc<TurnCancelResult>('turn.cancel', { session_key: sessionKey })
  }

  // ── Terminal ───────────────────────────────────────────────────────

  /** Report the terminal size on SIGWINCH. This is the only method that
   *  carries a width; the handler records it for the Rich console that
   *  `cli.dispatch` runs CLI commands against. Fire-and-forget. */
  terminalResize(cols: number, rows: number): Promise<TerminalResizeResult> {
    return this.transport.rpc<TerminalResizeResult>('terminal.resize', { cols, rows })
  }

  // ── Editor completion ──────────────────────────────────────────────

  completePath(word: string): Promise<CompletePathResult> {
    return this.transport.rpc<CompletePathResult>('complete.path', { word })
  }

  // ── Slash commands ─────────────────────────────────────────────────

  /** `command` is the text after the leading slash, as the handler expects. */
  slashExec(command: string, sessionId: null | string): Promise<SlashExecResult> {
    return this.transport.rpc<SlashExecResult>('slash.exec', { command, session_id: sessionId })
  }

  // ── Session management ─────────────────────────────────────────────

  sessionStatus(sessionId: string): Promise<SessionStatusResult> {
    return this.transport.rpc<SessionStatusResult>('session.status', { session_id: sessionId })
  }

  /** With `title`, sets it; without, reads the current one. */
  sessionTitle(sessionId: string, title?: string): Promise<SessionTitleResult> {
    return this.transport.rpc<SessionTitleResult>('session.title', {
      session_id: sessionId,
      ...(title === undefined ? {} : { title })
    })
  }

  sessionUndo(sessionId: string, n?: number): Promise<SessionUndoResult> {
    return this.transport.rpc<SessionUndoResult>('session.undo', {
      session_id: sessionId,
      ...(n === undefined ? {} : { n })
    })
  }

  sessionBranch(sessionId: null | string, name: string): Promise<SessionBranchResult> {
    return this.transport.rpc<SessionBranchResult>('session.branch', { name, session_id: sessionId })
  }

  sessionExport(sessionId: string): Promise<SessionExportResult> {
    return this.transport.rpc<SessionExportResult>('session.export', { session_id: sessionId })
  }

  /** The AGENTS.md / ODH.md files the next turn will send, and the toggle that
   *  takes one out of it for this session. */
  sessionInstructions(params: SessionInstructionsParams = {}): Promise<SessionInstructionsResult> {
    return this.transport.rpc<SessionInstructionsResult>('session.instructions', params)
  }

  sessionList(limit?: number): Promise<SessionListResult> {
    return this.transport.rpc<SessionListResult>('session.list', limit === undefined ? {} : { limit })
  }

  sessionDelete(sessionId: string): Promise<SessionDeleteResult> {
    return this.transport.rpc<SessionDeleteResult>('session.delete', { session_id: sessionId })
  }

  sessionClose(sessionId: string): Promise<SessionCloseResult> {
    return this.transport.rpc<SessionCloseResult>('session.close', { session_id: sessionId })
  }

  // ── Configuration ──────────────────────────────────────────────────

  /** Named keys only; `configGet()` fetches everything. */
  configGetKeys(keys: string[]): Promise<ConfigGetResult> {
    return this.transport.rpc<ConfigGetResult>('config.get', { keys })
  }

  // The handler accepts null before a session opens; the schema only declares string.
  configSet(params: Omit<ConfigSetParams, 'session_id'> & { session_id?: null | string }): Promise<ConfigSetResult> {
    return this.transport.rpc<ConfigSetResult>('config.set', params)
  }

  modelOverlay(
    field: ModelOverlayParams['field'],
    value: string,
    sessionId: null | string
  ): Promise<ModelOverlayResult> {
    return this.transport.rpc<ModelOverlayResult>('model.overlay', { field, session_id: sessionId, value })
  }

  reloadMcp(params: { always?: boolean; confirm?: boolean; session_id: null | string }): Promise<ReloadMcpResult> {
    return this.transport.rpc<ReloadMcpResult>('reload.mcp', params)
  }

  // ── Models and providers ───────────────────────────────────────────

  /** Providers, and their catalogues when `include_catalog` asks for them.
   *  Opening the picker asks for the cheap shape; entering a provider asks for
   *  that one slug's catalogue. */
  modelOptions(params: ModelOptionsParams): Promise<ModelOptionsResult> {
    return this.transport.rpc<ModelOptionsResult>('model.options', params)
  }

  /** Provider-wide, despite the session id: the credential is stored for the
   *  provider, not for this conversation. */
  modelSaveKey(params: ModelSaveKeyParams): Promise<ModelSaveKeyResult> {
    return this.transport.rpc<ModelSaveKeyResult>('model.save_key', params)
  }

  /** Writes a whole provider entry for an OpenAI-compatible endpoint: the wire
   *  is not a parameter, because that is the one this declares. */
  modelDeclareProvider(params: ModelDeclareProviderParams): Promise<ModelDeclareProviderResult> {
    return this.transport.rpc<ModelDeclareProviderResult>('model.declare_provider', params)
  }

  /** pi's scoped models: read the saved list, or write it with `write`. */
  modelScope(params: ModelScopeParams): Promise<ModelScopeResult> {
    return this.transport.rpc<ModelScopeResult>('model.scope', params)
  }

  /** pi's `/logout`: forgets the provider's credential and keeps what it declares. */
  modelLogout(params: ModelLogoutParams): Promise<ModelLogoutResult> {
    return this.transport.rpc<ModelLogoutResult>('model.logout', params)
  }

  /** Runs pi's own sign-in inside the gateway. Resolves when a credential is
   *  stored, which is minutes for a device code: every step arrives meanwhile as
   *  a `login.step` notification. */
  modelLogin(params: ModelLoginParams): Promise<ModelLoginResult> {
    return this.transport.rpc<ModelLoginResult>('model.login', params)
  }

  /** The answer a pushed step asked for: an option id, or a pasted code. */
  modelLoginAnswer(params: ModelLoginAnswerParams): Promise<ModelLoginAnswerResult> {
    return this.transport.rpc<ModelLoginAnswerResult>('model.login_answer', params)
  }

  modelLoginCancel(params: ModelLoginCancelParams): Promise<ModelLoginCancelResult> {
    return this.transport.rpc<ModelLoginCancelResult>('model.login_cancel', params)
  }

  // ── Broker replies ─────────────────────────────────────────────────

  /** Only the request's own id and conversation can resolve it. */
  approvalRespond(
    approvalId: string,
    conversationId: string,
    choice: 'allow' | 'deny'
  ): Promise<ApprovalRespondResult> {
    return this.transport.rpc<ApprovalRespondResult>('approval.respond', {
      approval_id: approvalId,
      choice,
      session_id: conversationId
    })
  }

  /** Keyed by the opaque request id alone. A `conversation_id` here takes
   *  precedence server-side and could answer a newer question, so it is never
   *  sent. */
  clarifyRespond(requestId: string, response: { answer: string } | { cancelled: true }): Promise<ClarifyRespondResult> {
    return this.transport.rpc<ClarifyRespondResult>('clarify.respond', { request_id: requestId, ...response })
  }

  confirmRespond(requestId: string, answer: boolean): Promise<ConfirmRespondResult> {
    return this.transport.rpc<ConfirmRespondResult>('confirm.respond', { answer, request_id: requestId })
  }
}
