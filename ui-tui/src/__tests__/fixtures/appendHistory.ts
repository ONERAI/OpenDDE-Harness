// A second process that appends prompts through the real `InputHistory`, for
// the concurrency tests. Run with tsx:
//
//   tsx fixtures/appendHistory.ts <file> <max> <entry>...
//
// Prints `APPENDED <n>` when every entry has been written.

import { InputHistory } from '../../lib/history.js'

const [file, max, ...entries] = process.argv.slice(2)

if (!file || !max) {
  throw new Error('usage: appendHistory.ts <file> <max> <entry>...')
}

const history = new InputHistory(file, Number(max))

history.load()

for (const entry of entries) {
  history.append(entry)
}

process.stdout.write(`APPENDED ${entries.length}\n`)
