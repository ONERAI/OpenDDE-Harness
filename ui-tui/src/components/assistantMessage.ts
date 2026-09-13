// One model reply in the transcript: thinking and text blocks in arrival
// order, and a closing note when the reply stopped for a reason worth saying
// out loud.
//
// Shaped after pi's `AssistantMessageComponent`: thinking collapses to a
// one-line label, and the message brackets itself with OSC 133 command-zone
// markers when no tool call follows it, so the terminal can fold the reply.

import { Container, MouseRegion, Spacer } from '@earendil-works/pi-tui'

import type { Theme } from '../theme.js'

import { StreamingMarkdown } from './streamingMarkdown.js'
import { ThemedText } from './themedText.js'

const OSC133_ZONE_START = '\x1b]133;A\x07'
const OSC133_ZONE_END = '\x1b]133;B\x07'
const OSC133_ZONE_FINAL = '\x1b]133;C\x07'

const THINKING_LABEL = 'Thinking…'

/** A block that grows by appending deltas and stops growing on `finish`. */
interface Block {
  append(delta: string): void
  finish(): void
}

/** Prose. `StreamingMarkdown` wants the whole text each time, so the block
 *  keeps it. */
class TextBlock extends StreamingMarkdown implements Block {
  private accumulated = ''

  get prose(): string {
    return this.accumulated
  }

  append(delta: string): void {
    this.accumulated += delta
    this.setText(this.accumulated)
  }
}

/** A reasoning run. Collapsed it is one dim line; expanded it is markdown in
 *  the same dim italic. Collapsing is what makes it cheap: the text still
 *  accumulates, but nothing parses it until someone asks to see it. */
class ThinkingBlock extends Container implements Block {
  private readonly body: StreamingMarkdown
  private readonly label: ThemedText

  private collapsed = true
  private text = ''

  constructor(theme: Theme, paddingX: number) {
    super()

    this.label = new ThemedText(text => theme.italic(theme.fg('muted', text)), THINKING_LABEL, paddingX, 0)
    this.body = new StreamingMarkdown('', paddingX, theme.markdownTheme(), {
      color: text => theme.fg('muted', text),
      italic: true
    })
    this.addChild(this.label)
  }

  append(delta: string): void {
    this.text += delta

    if (!this.collapsed) {
      this.body.setText(this.text)
    }
  }

  setCollapsed(collapsed: boolean): void {
    if (collapsed === this.collapsed) {
      return
    }

    this.collapsed = collapsed
    this.clear()

    if (collapsed) {
      this.addChild(this.label)
    } else {
      this.body.setText(this.text)
      this.addChild(this.body)
    }
  }

  finish(): void {
    this.body.finish()
  }
}

export interface AssistantMessageOptions {
  /** Left inset, in columns. pi uses 1. */
  paddingX?: number
  thinkingCollapsed?: boolean
}

export class AssistantMessage extends Container {
  private readonly blocks: Block[] = []
  private readonly content = new Container()
  private readonly paddingX: number
  private readonly theme: Theme

  private hasToolCalls = false
  private note: ThemedText | null = null
  private noteTone: 'error' | 'muted' = 'muted'
  private tail: 'text' | 'thinking' | null = null
  private thinkingCollapsed: boolean

  constructor(theme: Theme, opts: AssistantMessageOptions = {}) {
    super()

    this.theme = theme
    this.paddingX = opts.paddingX ?? 1
    this.thinkingCollapsed = opts.thinkingCollapsed ?? true

    this.addChild(this.content)
  }

  /** Nothing has been streamed into this message yet. */
  get isEmpty(): boolean {
    return this.blocks.length === 0 && this.note === null
  }

  /** The prose, without the thinking: what /copy, /retry and /history read. */
  get text(): string {
    return this.blocks
      .filter((block): block is TextBlock => block instanceof TextBlock)
      .map(block => block.prose)
      .join('\n\n')
  }

  appendText(delta: string): void {
    if (this.tail !== 'text') {
      const block = new TextBlock('', this.paddingX, this.theme.markdownTheme())

      this.open(block, block)
      this.tail = 'text'
    }

    this.blocks[this.blocks.length - 1].append(delta)
  }

  appendThinking(delta: string): void {
    if (this.tail !== 'thinking') {
      const block = new ThinkingBlock(this.theme, this.paddingX)

      block.setCollapsed(this.thinkingCollapsed)
      this.open(
        block,
        new MouseRegion(block, event => {
          if (event.type !== 'click' || event.button !== 'left') {
            return undefined
          }

          this.toggleThinking()

          return { handled: true }
        })
      )
      this.tail = 'thinking'
    }

    this.blocks[this.blocks.length - 1].append(delta)
  }

  /** A tool call follows, so this is not a terminal reply: skip the OSC 133
   *  command zone, which would close a block the turn is still writing. */
  setHasToolCalls(value: boolean): void {
    this.hasToolCalls = value
  }

  setThinkingCollapsed(collapsed: boolean): void {
    this.thinkingCollapsed = collapsed

    for (const block of this.blocks) {
      if (block instanceof ThinkingBlock) {
        block.setCollapsed(collapsed)
      }
    }
  }

  /** Returns true when thinking is now visible. */
  toggleThinking(): boolean {
    this.setThinkingCollapsed(!this.thinkingCollapsed)

    return !this.thinkingCollapsed
  }

  /** A closing line under the reply: stop reason, or the error that ended it. */
  setNote(text: string, tone: 'error' | 'muted' = 'muted'): void {
    this.noteTone = tone

    if (this.note) {
      this.note.setRaw(text)

      return
    }

    // The tone is read through the field so a later `setNote` retones the same
    // component, and the colour itself is resolved per render.
    this.note = new ThemedText(note => this.theme.fg(this.noteTone, note), text, this.paddingX, 0)
    this.content.addChild(new Spacer(1))
    this.content.addChild(this.note)
  }

  /** No more deltas. Freezes every block so none of them re-parses again. */
  finish(): void {
    for (const block of this.blocks) {
      block.finish()
    }

    this.tail = null
  }

  override render(width: number): string[] {
    const lines = super.render(width)

    if (this.hasToolCalls || lines.length === 0) {
      return lines
    }

    lines[0] = OSC133_ZONE_START + lines[0]
    lines[lines.length - 1] = OSC133_ZONE_END + OSC133_ZONE_FINAL + lines[lines.length - 1]

    return lines
  }

  /** Start a new block. `mounted` is what goes in the tree, which for thinking
   *  is the block wrapped in its click region. */
  private open(block: Block, mounted: Container | MouseRegion): void {
    this.content.addChild(new Spacer(1))
    this.content.addChild(mounted)
    this.blocks.push(block)
  }
}
