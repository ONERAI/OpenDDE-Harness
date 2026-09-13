// What a request carries as its output ceiling when nobody sized the model.
//
// pi sends the ceiling it is given, or the model's own, clamped to the window
// (`clampMaxTokensToContext`): with a window it knows, a zero ceiling comes
// out as zero and the OpenAI-shaped wires then send none at all, so the
// server's own default applies; with no window either, the zero is floored at
// one token, and a one-token reply is what a request for a model with neither
// limit used to get. So:
//
// * a ceiling the request names is sent as named, always;
// * on an OpenAI-shaped wire, a model with neither limit is streamed under an
//   unbounded window for the clamp's sake only, which makes pi omit the
//   ceiling -- the same request every OpenAI-compatible client sends for a
//   model it has not sized, and what a relay's own `/models` row leaves us
//   with. Nothing reported about the model changes: `models` still says the
//   window is unknown, and the trimmer keeps treating it that way;
// * on the other wires the zero would travel as the ceiling and be refused by
//   the vendor, so the request is refused here, naming the fix.

import type { Api, Model } from '@earendil-works/pi-ai'

/** The wires that omit the ceiling when it is falsy (`if (options?.maxTokens)`). */
const OMITS_FALSY_CEILING = new Set<string>([
  'azure-openai-responses',
  'openai-codex-responses',
  'openai-completions',
  'openai-responses'
])

/** A window so large that the clamp leaves a zero ceiling at zero. */
export const UNBOUNDED_WINDOW = Number.MAX_SAFE_INTEGER

export interface CeilingDecision {
  /** The model to hand pi: the row itself, or a copy with an unbounded window. */
  model: Model<Api>
  /** Why the request cannot be sent, when it cannot. */
  refusal?: string
}

export function ceilingFor(model: Model<Api>, requested: number | undefined): CeilingDecision {
  if ((requested ?? 0) > 0 || model.maxTokens > 0 || model.contextWindow > 0) {
    return { model }
  }
  if (OMITS_FALSY_CEILING.has(model.api)) {
    return { model: { ...model, contextWindow: UNBOUNDED_WINDOW } }
  }
  return {
    model,
    refusal:
      `${model.provider}/${model.id} declares no maxTokens and no contextWindow, and ${model.api} sends a ` +
      'zero ceiling as written; declare maxTokens for this model (ddeharness provider set), or name one on the request'
  }
}
