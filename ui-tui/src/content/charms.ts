// Wording for the reassurance row a tool panel grows when a call takes a long
// time.
//
// The old TUI pushed "still folding…" into the transcript for any tool that ran
// past eight seconds, which said a protein was being folded whatever the tool
// was actually doing. The domain phrases are kept, but each one now has to be
// earned by the tool's own name or by what its progress line says; everything
// else gets neutral wording. The row always carries the elapsed time, because
// that is the only thing about it the UI actually knows.

export const GENERIC_WAIT = ['still working', 'still running'] as const

interface DomainWait {
  match: RegExp
  phrases: readonly string[]
}

// The three domain phrases are the old TUI's own wording, unchanged; what is
// new is that each one has to be earned.
const DOMAIN_WAIT: readonly DomainWait[] = [
  { match: /fold|boltz|alphafold|esmfold|structure/i, phrases: ['still folding'] },
  { match: /score|rank|objective|candidate|affinity/i, phrases: ['scoring candidates'] },
  { match: /compute|gpu|lease|worker|remote|cluster/i, phrases: ['waiting on the compute service'] }
]

/** Phrases allowed for this call, most specific first. */
export function waitPhrases(toolName: string, progress = ''): readonly string[] {
  const context = `${toolName} ${progress}`
  const matched = DOMAIN_WAIT.filter(entry => entry.match.test(context)).flatMap(entry => entry.phrases)

  return matched.length > 0 ? matched : GENERIC_WAIT
}

/** `still folding… · waiting 8s`. */
export function waitLine(phrase: string, elapsedMs: number): string {
  return `${phrase}… · waiting ${Math.round(elapsedMs / 1000)}s`
}
