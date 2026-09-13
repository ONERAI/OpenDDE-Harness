// The benchmark replies the streaming work was measured against, so the tests
// assert byte-exactness on the same text the numbers came from. Copied from
// `<scratchpad>/tui-bench/stream.ts`, trimmed to the uniform chunking the
// acceptance suite uses.

const PARA = (n: number): string =>
  `Paragraph ${n}: the reducer keeps the parsed prefix and only re-tokenizes the in-flight tail, so a long reply does not cost O(n) per token. ` +
  `It also coalesces deltas into frame-sized batches so the terminal repaints at most once per frame. `

const codeLine = (i: number): string =>
  `export const fn${i} = (x: number): number => { const y = x * ${i}; return y + "s".length } // line ${i}`

const CODE = ['```ts', ...Array.from({ length: 40 }, (_, i) => codeLine(i)), '```'].join('\n')

const TABLE = [
  '| name | count | status |',
  '|---|---|---|',
  ...Array.from({ length: 12 }, (_, i) => `| item-${i} | ${i * 7} | ${i % 2 ? 'ok' : 'pending'} |`)
].join('\n')

const buildText = (): string =>
  [
    '# Streaming benchmark',
    '',
    PARA(1),
    '',
    '## Setup',
    '',
    '- first bullet with **bold** and `code`',
    '- second bullet with a [link](https://example.com)',
    '- third bullet',
    '',
    PARA(2),
    '',
    CODE,
    '',
    PARA(3),
    '',
    TABLE,
    '',
    PARA(4),
    '',
    '> a quote that spans a while and keeps going for a bit longer than one line of the terminal',
    '',
    PARA(5),
    '',
    '```py',
    ...Array.from({ length: 30 }, (_, i) => `def f${i}(x):\n    return x + ${i}  # comment`),
    '```',
    '',
    PARA(6),
    PARA(7),
    '',
    PARA(8),
    '',
    '## Second half',
    '',
    PARA(9),
    '',
    ...Array.from({ length: 10 }, (_, i) => `${i + 1}. numbered item ${i} with some trailing words to wrap the line`),
    '',
    PARA(10),
    '',
    '```sh',
    ...Array.from({ length: 25 }, (_, i) => `echo "step ${i}" && ls -la /tmp/dir${i} | grep -c foo # ${i}`),
    '```',
    '',
    PARA(11),
    PARA(12),
    '',
    TABLE,
    '',
    PARA(13),
    '',
    PARA(14)
  ].join('\n')

export type StreamKind = 'default' | 'fence' | 'long'

export function streamText(kind: StreamKind): string {
  if (kind === 'long') {
    return Array.from({ length: 4 }, () => buildText()).join('\n\n')
  }

  if (kind === 'fence') {
    return ['```ts', ...Array.from({ length: 300 }, (_, i) => codeLine(i)), '```'].join('\n')
  }

  return buildText()
}

/** `count` deltas of equal length, the uniform chunking the benchmark used. */
export function buildStream(kind: StreamKind, count = 2000): string[] {
  const text = streamText(kind)

  return Array.from({ length: count }, (_, i) =>
    text.slice(Math.floor((i * text.length) / count), Math.floor(((i + 1) * text.length) / count))
  )
}
