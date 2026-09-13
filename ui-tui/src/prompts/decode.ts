// What arrives on the notification socket, checked before anything is drawn or
// answered.
//
// The generated DTOs describe the contract, not the bytes: a malformed frame
// still type-checks as `unknown`. A prompt with a missing id cannot be answered,
// and a `default` that is not a real boolean must never be run through
// `Boolean()`, so each decoder returns null and the frame is dropped.
//
// `fields<T>` reads the payload as unknown values under the generated field
// names, so renaming one in the schema breaks this file rather than silently
// producing empty prompts.

import type {
  ApprovalClosedNotification,
  ApprovalRequestNotification,
  ClarifyRequestNotification,
  ConfirmRequestNotification
} from '../rpc/index.js'

export interface ApprovalRequest {
  actionDigest: string
  approvalId: string
  command: string
  conversationId: string
  createdAt: number
  description: string
  /** Unix **seconds**, as the broker serializes it. */
  expiresAt: number
  toolCallId: string
  turnId: string
}

export interface ApprovalClosed {
  approvalId: string
  conversationId: string
  reason: string
}

export interface ClarifyRequest {
  choices: string[]
  conversationId: string
  question: string
  requestId: string
}

export interface ConfirmRequest {
  defaultAnswer: boolean
  prompt: string
  requestId: string
}

/** The payload as unknown values under the names the schema declares. */
function fields<T>(params: unknown): null | Partial<Record<keyof T, unknown>> {
  return typeof params === 'object' && params !== null && !Array.isArray(params)
    ? (params as Partial<Record<keyof T, unknown>>)
    : null
}

function text(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function id(value: unknown): null | string {
  return typeof value === 'string' && value.length > 0 ? value : null
}

function seconds(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

export function decodeApprovalRequest(params: unknown): ApprovalRequest | null {
  const p = fields<ApprovalRequestNotification['params']>(params)

  if (!p) {
    return null
  }

  const approvalId = id(p.approval_id)
  const conversationId = id(p.conversation_id)

  // Both are required to answer: `approval.respond` is keyed by the pair, and a
  // request that cannot be answered must not be shown as if it could be.
  if (!approvalId || !conversationId) {
    return null
  }

  return {
    actionDigest: text(p.action_digest),
    approvalId,
    command: text(p.command),
    conversationId,
    createdAt: seconds(p.created_at),
    description: text(p.description) || 'a command needs approval',
    expiresAt: seconds(p.expires_at),
    toolCallId: text(p.tool_call_id),
    turnId: text(p.turn_id)
  }
}

export function decodeApprovalClosed(params: unknown): ApprovalClosed | null {
  const p = fields<ApprovalClosedNotification['params']>(params)

  if (!p) {
    return null
  }

  const approvalId = id(p.approval_id)
  const conversationId = id(p.conversation_id)

  if (!approvalId || !conversationId) {
    return null
  }

  return { approvalId, conversationId, reason: text(p.reason) || 'closed' }
}

export function decodeClarifyRequest(params: unknown): ClarifyRequest | null {
  const p = fields<ClarifyRequestNotification['params']>(params)

  if (!p) {
    return null
  }

  const requestId = id(p.request_id)

  if (!requestId) {
    return null
  }

  const choices = Array.isArray(p.choices)
    ? p.choices.filter((choice): choice is string => typeof choice === 'string')
    : []

  return {
    choices,
    conversationId: text(p.conversation_id),
    question: text(p.question) || 'The agent asked a question.',
    requestId
  }
}

export function decodeConfirmRequest(params: unknown): ConfirmRequest | null {
  const p = fields<ConfirmRequestNotification['params']>(params)

  if (!p) {
    return null
  }

  const requestId = id(p.request_id)

  // A default that is not a boolean is not a default: the broker applies the
  // real one on timeout, and guessing here would answer for the user.
  if (!requestId || typeof p.default !== 'boolean') {
    return null
  }

  return {
    defaultAnswer: p.default,
    prompt: text(p.prompt) || 'Confirm?',
    requestId
  }
}
