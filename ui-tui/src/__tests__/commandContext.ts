// A CommandContext backed by spies, for the command tests.

import { vi } from 'vitest'

import type { CommandContext, RetryCandidate, SessionInfoView, TranscriptEntry } from '../commands/index.js'
import type { BusyInputMode } from '../components/queue.js'
import type { SystemTone } from '../components/systemLine.js'
import type { CommandsCatalogResult } from '../rpc/index.js'

import { Gateway } from '../gateway.js'
import { createKeybindings } from '../lib/keybindings.js'
import { ProteinDesignTasks } from '../tasks/monitor.js'
import { FakeTransport } from './fakes.js'
import { ManualClock } from './selectorHarness.js'

export interface PrintedLine {
  text: string
  title?: string
  tone?: SystemTone
}

export interface TestContext {
  /** The live queue mode, as the app's would be. */
  busyMode: BusyInputMode
  context: CommandContext
  info: SessionInfoView
  /** Whether collapsed tool rows are quiet, as `/quiet-tools` left it. */
  quietTools: boolean
  printed: PrintedLine[]
  /** What `/retry` resends, as the app would have recorded it. */
  retry: null | RetryCandidate
  sessionId: null | string
  /** The real monitor, over whatever root the test asked for. */
  tasks: ProteinDesignTasks
  spies: {
    adopt: ReturnType<typeof vi.fn>
    create: ReturnType<typeof vi.fn>
    dropLastExchange: ReturnType<typeof vi.fn>
    openLoginPicker: ReturnType<typeof vi.fn>
    openLogoutPicker: ReturnType<typeof vi.fn>
    openModelPicker: ReturnType<typeof vi.fn>
    openScopedModelsPicker: ReturnType<typeof vi.fn>
    openThinkingPicker: ReturnType<typeof vi.fn>
    setShowTokenUsage: ReturnType<typeof vi.fn>
    showHelp: ReturnType<typeof vi.fn>
    pick: ReturnType<typeof vi.fn>
    quit: ReturnType<typeof vi.fn>
    remove: ReturnType<typeof vi.fn>
    resume: ReturnType<typeof vi.fn>
    send: ReturnType<typeof vi.fn>
    setBusyMode: ReturnType<typeof vi.fn>
  }
  transcript: TranscriptEntry[]
  transport: FakeTransport
  /** Everything printed, joined, for substring assertions. */
  text(): string
}

export function createTestContext(
  results: Record<string, unknown> = {},
  options: {
    busy?: boolean
    busyMode?: BusyInputMode
    catalog?: CommandsCatalogResult | null
    retry?: null | RetryCandidate
    sessionId?: null | string
    /** A temporary directory of task fixtures. Defaults to one that is not
     *  there, which is the same as having started no designs. */
    taskRoot?: string
  } = {}
): TestContext {
  const transport = new FakeTransport(results)
  const printed: PrintedLine[] = []
  const transcript: TranscriptEntry[] = []
  const info: SessionInfoView = {}
  /** `/quiet-tools`, as the app's tool panels would hold it. */
  const quiet = { tools: true }

  const state = {
    busyMode: options.busyMode ?? ('queue' as BusyInputMode),
    epoch: 1,
    retry: options.retry ?? null,
    sessionId: options.sessionId === undefined ? 'tui:abc' : options.sessionId,
    showTokenUsage: true
  }

  const tasks = new ProteinDesignTasks({
    clock: new ManualClock(),
    root: options.taskRoot ?? '/nonexistent/opendde-harness-test-task-root'
  })

  const spies = {
    adopt: vi.fn(async (id: string, title?: string) => {
      state.sessionId = id

      if (title) {
        info.title = title
      }
    }),
    create: vi.fn(async () => {}),
    dropLastExchange: vi.fn(() => {
      // The app removes the last *saved* exchange; the fake transcript has no
      // unsaved echoes, so its last pair is that exchange.
      if (transcript.length === 0) {
        return false
      }

      transcript.splice(-2, 2)

      if (state.retry) {
        state.retry = { ...state.retry, onServer: false }
      }

      return true
    }),
    openLoginPicker: vi.fn(),
    openLogoutPicker: vi.fn(),
    openModelPicker: vi.fn(),
    openScopedModelsPicker: vi.fn(),
    openThinkingPicker: vi.fn(),
    setShowTokenUsage: vi.fn(async (_value: boolean) => true),
    showHelp: vi.fn(),
    pick: vi.fn(),
    quit: vi.fn(),
    remove: vi.fn(async () => true),
    resume: vi.fn(async () => {}),
    send: vi.fn(),
    setBusyMode: vi.fn((mode: BusyInputMode) => {
      state.busyMode = mode
    })
  }

  const context: CommandContext = {
    busyMode: () => state.busyMode,
    catalog: () => options.catalog ?? null,
    gateway: new Gateway(transport),
    keybindings: createKeybindings(),
    openLoginPicker: spies.openLoginPicker,
    openLogoutPicker: spies.openLogoutPicker,
    openModelPicker: spies.openModelPicker,
    openScopedModelsPicker: spies.openScopedModelsPicker,
    openThinkingPicker: spies.openThinkingPicker,
    quietTools: {
      get: () => quiet.tools,
      set: value => {
        quiet.tools = value
      }
    },
    quit: spies.quit,
    session: {
      adopt: spies.adopt,
      blockedReason: () => (options.busy ? 'a turn is running' : null),
      busy: () => options.busy ?? false,
      create: spies.create,
      id: () => state.sessionId,
      info: () => info,
      patchInfo: patch => Object.assign(info, patch),
      pick: spies.pick,
      remove: spies.remove,
      resume: spies.resume,
      send: spies.send
    },
    setBusyMode: spies.setBusyMode,
    setShowTokenUsage: spies.setShowTokenUsage,
    showHelp: spies.showHelp,
    showTokenUsage: () => state.showTokenUsage,
    tasks: {
      focus: taskId => tasks.focus(taskId),
      list: () => tasks.list(),
      readLogTail: (taskId, maxLines) => tasks.readLogTail(taskId, maxLines),
      refresh: () => tasks.refresh(),
      resolve: query => tasks.resolve(query),
      rootFailure: () => tasks.rootFailure,
      rootPath: () => tasks.rootPath,
      sessionEpoch: () => state.epoch,
      setVisible: visible => tasks.setVisible(visible),
      visible: () => tasks.visible
    },
    transcript: {
      dropLastExchange: spies.dropLastExchange,
      history: () => transcript,
      print: (text, tone) => {
        printed.push({ text, ...(tone ? { tone } : {}) })
      },
      printBlock: (text, title) => {
        printed.push({ text, ...(title ? { title } : {}) })
      },
      retryCandidate: () => state.retry
    }
  }

  return {
    get busyMode() {
      return state.busyMode
    },
    context,
    get quietTools() {
      return quiet.tools
    },
    info,
    printed,
    get retry() {
      return state.retry
    },
    get sessionId() {
      return state.sessionId
    },
    spies,
    tasks,
    text: () => printed.map(line => line.text).join('\n'),
    transcript,
    transport
  }
}
