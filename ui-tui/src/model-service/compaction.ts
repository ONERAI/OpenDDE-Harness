// Codex-style server-side compaction, over the same pipe as everything else.
//
// The algorithms are the owner's pi extension, used as a library and not
// copied: `pi-codex-compact` (pinned by commit in package.json) converts pi
// messages to Responses items, asks the Responses API for an opaque
// replacement history through `compaction_trigger`, and patches a later
// request to replay that history. What we replace is its `src/index.ts` --
// pi's compaction lifecycle -- because here the Python loop decides when to
// compact and what to replay. Its two hooks become two request paths:
//
//   session_before_compact   -> `compact`  (runCompaction)
//   before_provider_request  -> `stream` with `replay` (replayOnPayload)
//
// Deliberately not carried over: the extension's local text summary (the
// Python side writes its own), its `notify` UI, and the per-session replay
// state it reconstructs from pi's session file -- the caller holds that and
// sends it back as `replay`.

import type { Api, Context, Model, Models, SimpleStreamOptions, Tool, Usage } from '@earendil-works/pi-ai'
import type { ResponseItem } from 'pi-codex-compact/src/compaction.ts'

import {
  buildRemoteCompactionV2History,
  buildToolsPayload,
  callRemoteCompactionEndpoint,
  messagesToResponseItems,
  normalizeResponseItemsForPrompt
} from 'pi-codex-compact/src/compaction.ts'
import {
  applyRemoteHistoryPayloadPatch,
  looksLikeResponsesPayload,
  supportsRemoteCompactionModel,
  thinkingLevelToResponsesReasoning
} from 'pi-codex-compact/src/openai.ts'

import type { CompactParams, ReplayHistory } from './protocol.js'

import { BadRequest } from './protocol.js'

export type { ResponseItem }

export interface CompactionOutcome {
  items: ResponseItem[]
  usage?: Usage
}

/** The two `Models` reads this path makes, named so a test can hand it a stub. */
export type CompactionModels = Pick<Models, 'getAuth' | 'getModel'>

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** pi lets a credential suppress a provider default with a null header; our own fetch has none to suppress. */
function plainHeaders(headers: Record<string, string | null> | undefined): Record<string, string> | undefined {
  if (!headers) {
    return undefined
  }
  return Object.fromEntries(Object.entries(headers).filter(([, value]) => typeof value === 'string')) as Record<
    string,
    string
  >
}

/**
 * The prompt the model sees: the replayed history, then every message since.
 * This is what the extension keeps as `explicitHistory`, rebuilt per request
 * because the caller -- not this process -- owns the session.
 */
function replayedInput(replay: ReplayHistory | undefined, context: Context): ResponseItem[] {
  const live = messagesToResponseItems(context.messages)
  return replay && replay.items.length > 0 ? [...replay.items, ...live] : live
}

function toolsPayload(tools: Tool[] | undefined): Record<string, unknown>[] {
  const declared = tools ?? []
  return buildToolsPayload(
    declared,
    declared.map(tool => tool.name)
  )
}

/**
 * The model to compact with, or the refusal. Everything but an OpenAI
 * Responses or Codex Responses model is refused here: nothing else serves
 * this endpoint, and the extension gates its replay on the same question.
 */
function compactionModel(models: CompactionModels, params: CompactParams): Model<Api> {
  const model = models.getModel(params.provider, params.model)
  if (!model) {
    throw new BadRequest('model_not_found', `no model ${params.provider}/${params.model}`)
  }
  if (!supportsRemoteCompactionModel(model)) {
    throw new BadRequest(
      'unsupported_model',
      `${params.provider}/${params.model} has no server-side compaction: it is not an OpenAI Responses model`
    )
  }
  return model
}

/**
 * Ask the Responses API to compact the conversation.
 *
 * `fetch` is the global one -- the extension's endpoint call takes no injected
 * fetch -- so a test stubs `globalThis.fetch`.
 */
export async function runCompaction(
  models: CompactionModels,
  params: CompactParams,
  signal?: AbortSignal
): Promise<CompactionOutcome> {
  const model = compactionModel(models, params)
  const auth = await models.getAuth(model, { signal })
  const apiKey = auth?.auth.apiKey
  if (!apiKey) {
    throw new BadRequest('auth', `Provider is not configured: ${params.provider}`)
  }
  // A credential may name the endpoint; pi sends the turn there, so compaction
  // follows it (`models.ts` builds the same request model on the stream path).
  const requestModel: Model<Api> = auth?.auth.baseUrl ? { ...model, baseUrl: auth.auth.baseUrl } : model
  const result = await callRemoteCompactionEndpoint({
    apiKey,
    headers: plainHeaders(auth?.auth.headers),
    input: normalizeResponseItemsForPrompt(replayedInput(params.replay, params.context), requestModel),
    instructions: params.context.systemPrompt,
    model: requestModel,
    parallelToolCalls: true,
    reasoning: requestModel.reasoning ? thinkingLevelToResponsesReasoning(params.reasoning) : undefined,
    sessionId: params.sessionId,
    signal,
    tools: toolsPayload(params.tools ?? params.context.tools)
  })
  return { items: result.output, usage: result.usage }
}

/**
 * The `onPayload` hook that replays a replacement history, or undefined when
 * there is nothing to replay.
 *
 * pi calls this with the provider payload it is about to send and takes the
 * returned object as a replacement (`undefined` keeps the payload). The model
 * gate is the extension's own and load-bearing: `looksLikeResponsesPayload`
 * alone says yes to any payload with a `model` or `messages` key, which a
 * chat-completions request also has.
 */
export function replayOnPayload(
  model: Model<Api>,
  replay: ReplayHistory | undefined,
  context: Context
): SimpleStreamOptions['onPayload'] {
  if (!replay || replay.items.length === 0 || !supportsRemoteCompactionModel(model)) {
    return undefined
  }
  return (payload: unknown) => {
    if (!isRecord(payload) || !looksLikeResponsesPayload(payload)) {
      return undefined
    }
    return applyRemoteHistoryPayloadPatch({
      explicitHistory: normalizeResponseItemsForPrompt(replayedInput(replay, context), model),
      payload
    })
  }
}

// --- faux mode (OPENDDE_MODEL_SERVICE_FAUX=1) -------------------------------
// Offline stand-ins. Nothing below is reachable unless the caller asked for a
// faux provider, and none of it touches the network.

/** The shape of a real result -- retained user messages, then the opaque item -- with a scripted payload. */
export function fauxCompaction(models: CompactionModels, params: CompactParams, sequence: number): CompactionOutcome {
  // Refused for the same models the real path refuses: a faux provider that is
  // not shaped like a Codex one has no more compaction than a chat model does.
  compactionModel(models, params)
  const input = replayedInput(params.replay, params.context)
  const items = buildRemoteCompactionV2History(input, {
    encrypted_content: `faux-${sequence}`,
    type: 'compaction'
  })
  return {
    items,
    usage: {
      cacheRead: 0,
      cacheWrite: 0,
      cost: { cacheRead: 0, cacheWrite: 0, input: 0, output: 0, total: 0 },
      input: input.length,
      output: 1,
      totalTokens: input.length + 1
    }
  }
}

/**
 * A Responses request as pi would build one, for the faux stream path.
 *
 * pi's faux provider never calls `onPayload` -- it builds no payload at all --
 * so the service runs the hook against this to let a test see that the replay
 * reached the request.
 */
export function fauxResponsesPayload(model: Model<Api>, context: Context): Record<string, unknown> {
  return {
    input: messagesToResponseItems(context.messages),
    instructions: context.systemPrompt,
    model: model.id,
    previous_response_id: 'faux-previous-response',
    store: false,
    stream: true
  }
}
