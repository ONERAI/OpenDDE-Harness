/** Choosing one of a fixed set of phrases. `random` is injected by the tests
 *  that assert on wording, so nothing has to reach for a global seed. */
export function pick<T>(items: readonly T[], random: () => number = Math.random): T {
  return items[Math.floor(random() * items.length)] ?? items[0]!
}
