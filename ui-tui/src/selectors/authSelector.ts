// pi's `/login` and `/logout`: the authentication method, then the provider,
// then the sign-in — or the provider to sign out of.
//
// Stages of one selector, not one selector each: going back rebuilds the body
// in place, so the slot changes hands once and the chat editor underneath keeps
// its draft the whole time. The credential, endpoint and sign-in stages live in
// `modelStages.ts`; this is the two doors and the two lists behind them.
//
// Models are not chosen here and providers are not listed for their models:
// that is `/model`, and it lists what the connected providers serve. Connecting
// a provider is this selector's whole job, as it is pi's `/login`'s.

import type { Selector, SelectorLease } from '../components/selector.js'
import type { SelectorRow } from '../components/selectorList.js'
import type { ModelOptionProvider } from '../rpc/index.js'
import type { SelectorDeps } from './deps.js'
import type { ModelStageContext } from './modelStages.js'

import { SelectorList } from '../components/selectorList.js'
import { StageView } from '../components/stage.js'
import {
  ConfigFieldReadonlyError,
  ConfigValidationError,
  ModelNotAvailableError,
  NotSupportedInV01Error
} from '../rpc/index.js'
import { selectorError } from './deps.js'
import { createModelStages, SIGN_IN_WITH_ACCOUNT, SIGN_IN_WITH_API_KEY } from './modelStages.js'

/** The row under the API-key method for a provider no gateway knows of yet. */
const DECLARE_ENDPOINT = 'action:declare-endpoint'

/** Which door: pi's `/login`, or pi's `/logout`. */
export type AuthEntry = 'login' | 'logout'

/** pi's own two method names as its selector rows are keyed. */
type AuthMethod = 'key' | 'oauth'

export interface AuthSelectorOptions {
  deps: SelectorDeps
  entry: AuthEntry
  /** `/login <provider>`: the name or id typed after the command, if any. */
  entryProvider?: string
  lease: SelectorLease
}

/** What `-32008` and friends should say. The raw `data.errors` is never shown:
 *  model validation echoes its input, which can include a key just typed. */
function explain(err: unknown): string {
  if (err instanceof ModelNotAvailableError) {
    return 'that model is not available — the provider may need a key or a sign-in first'
  }

  if (err instanceof ConfigValidationError) {
    return 'the gateway rejected that value'
  }

  if (err instanceof NotSupportedInV01Error) {
    return 'this gateway build cannot do that yet — use the ddeharness CLI'
  }

  if (err instanceof ConfigFieldReadonlyError) {
    return 'the gateway does not accept that change'
  }

  return selectorError(err)
}

export function createAuthSelector({ deps, entry, entryProvider, lease }: AuthSelectorOptions): Selector {
  const { theme } = deps
  const stage = new StageView()
  const sessionId = lease.sessionId

  let providers: ModelOptionProvider[] = []
  /** Bumped by every request, so an older answer can never paint. */
  let sequence = 0

  const stageContext: ModelStageContext = {
    back: message => showLoginProviders(undefined, undefined, message),
    deps,
    explain,
    finish: message => {
      deps.transcript.print(message)
      lease.done('done')
    },
    lease,
    provider: slug => provider(slug),
    sessionId,
    stage
  }
  const stages = createModelStages(stageContext)

  // ── data ───────────────────────────────────────────────────────────

  async function loadOptions(): Promise<void> {
    const mine = ++sequence
    const result = await deps.gateway.modelOptions({ include_catalog: false, session_id: sessionId })

    if (!lease.isCurrent() || mine !== sequence) {
      return
    }

    providers = result.providers ?? []
  }

  function provider(slug: string): ModelOptionProvider | undefined {
    return providers.find(item => item.slug === slug)
  }

  // ── pi's own /login ────────────────────────────────────────────────

  /**
   * pi's own status marker for one row (`oauth-selector.formatStatusIndicator`).
   *
   * The words are pi's: nothing stored is "unconfigured", a credential of this
   * very method is "configured" (or the variable that supplies it), and a
   * provider configured by its *other* method says which — signing in to a
   * provider that already holds a key is a real thing to do, and the row has to
   * say what is there now.
   */
  function statusMarker(item: ModelOptionProvider, method: AuthMethod): string {
    if (!item.authenticated) {
      return '• unconfigured'
    }

    if (item.auth_type !== method) {
      return item.auth_type === 'oauth' ? '• subscription configured' : '• API key configured'
    }

    return item.key_env ? `✓ env: ${item.key_env}` : '✓ configured'
  }

  /** pi's own label for a method, for a list that mixes both. */
  function methodLabel(method: AuthMethod): string {
    return method === 'oauth' ? 'subscription' : 'API key'
  }

  /**
   * Which ways in one row offers, in pi's order.
   *
   * `auth_methods` is pi's own list, read off its provider objects by the model
   * service. A gateway that could not ask it sends none, and then the row's own
   * `auth_type` decides — the one way in the gate would judge a submission by —
   * so a machine whose model service is not runnable yet can still be handed a
   * key. The same rule the wizard applies (`providers.login_flow.login_methods`).
   */
  function loginMethods(item: ModelOptionProvider): AuthMethod[] {
    const own = item.auth_methods.filter((method): method is AuthMethod => method === 'oauth' || method === 'key')

    if (own.length > 0) {
      return own
    }

    return item.auth_type === 'oauth' || item.auth_type === 'key' ? [item.auth_type] : []
  }

  /** Every (provider, method) pair pi would offer, in pi's own order: by name. */
  function loginPairs(method?: AuthMethod): { item: ModelOptionProvider; method: AuthMethod }[] {
    const pairs: { item: ModelOptionProvider; method: AuthMethod }[] = []

    for (const item of [...providers].sort((a, b) => a.name.localeCompare(b.name))) {
      for (const candidate of loginMethods(item)) {
        if (!method || candidate === method) {
          pairs.push({ item, method: candidate })
        }
      }
    }

    return pairs
  }

  /**
   * pi's "Select provider to configure:" — one row per way in, as pi lists them.
   *
   * A provider that offers both appears twice when the list is not filtered to
   * one method, which is pi's own shape: the row *is* the choice, so there is
   * nothing left to ask afterwards. pi shows the method label only when the list
   * holds more than one kind, and so does this.
   */
  function showLoginProviders(method?: AuthMethod, query?: string, message?: string): void {
    const pairs = loginPairs(method)
    const showMethod = new Set(pairs.map(pair => pair.method)).size > 1
    const rows: SelectorRow[] = pairs.map(({ item, method: pairMethod }) => ({
      description: `${item.slug}  ·  ${statusMarker(item, pairMethod)}`,
      id: `${item.slug}:${pairMethod}`,
      kind: 'item' as const,
      label: `${item.name}${showMethod ? ` [${methodLabel(pairMethod)}]` : ''}`,
      searchText: `${item.name} ${item.slug} ${pairMethod} ${item.login_label ?? ''} ${item.key_label ?? ''}`
    }))

    // Ours, and the one thing here that is not pi's: an endpoint pi has never
    // heard of is reached by a key, so it belongs under the key method.
    if (method !== 'oauth') {
      rows.push({
        description: 'custom  ·  key',
        id: DECLARE_ENDPOINT,
        kind: 'action',
        label: 'OpenAI Compatible',
        searchText: 'openai compatible custom endpoint relay base url vllm'
      })
    }

    const list = new SelectorList({
      emptyText: 'No providers available',
      keybindings: deps.keybindings,
      onCancel: () => (method && entry === 'login' ? showAuthMethodMenu() : lease.done('cancel')),
      onSelect: id => {
        if (id === DECLARE_ENDPOINT) {
          stages.showDeclareEndpoint()

          return
        }

        const [slug = '', chosen = ''] = id.split(':')

        if (chosen === 'oauth') {
          stages.showLogin(slug)
        } else {
          stages.showKeyForm(slug)
        }
      },
      placeholder: 'search providers',
      rows,
      theme,
      // pi's own title for this step.
      title: 'Select provider to configure:',
      tui: deps.tui
    })

    if (query) {
      list.setQuery(query)
    }

    if (message) {
      list.setError(message)
    }

    stage.set(list)
    deps.tui.requestRender()
  }

  /** pi's first step for a bare `/login`: which way in, before which provider. */
  function showAuthMethodMenu(): void {
    const list = new SelectorList({
      keybindings: deps.keybindings,
      onCancel: () => lease.done('cancel'),
      onSelect: id => showLoginProviders(id as AuthMethod),
      placeholder: 'search methods',
      rows: [
        {
          description: 'a provider you are subscribed to',
          id: 'oauth',
          kind: 'item',
          label: SIGN_IN_WITH_ACCOUNT,
          searchText: `${SIGN_IN_WITH_ACCOUNT} oauth subscription account sign in`
        },
        {
          description: 'a key you paste, or an endpoint of your own',
          id: 'key',
          kind: 'item',
          label: SIGN_IN_WITH_API_KEY,
          searchText: `${SIGN_IN_WITH_API_KEY} api key endpoint`
        }
      ],
      theme,
      // pi's own title, and pi's own two options in pi's own order.
      title: 'Select authentication method:',
      tui: deps.tui
    })

    stage.set(list)
    deps.tui.requestRender()
  }

  /** `/login <provider>`: pi goes straight to the one it names. */
  function enterNamedProvider(typed: string): void {
    const wanted = typed.trim().toLowerCase()
    const match = providers.find(item => item.slug.toLowerCase() === wanted || item.name.toLowerCase() === wanted)

    if (match && loginMethods(match).length > 0) {
      // One way in goes straight to it; both ask pi's question first.
      stages.showAuthMethod(match.slug, () => showLoginProviders())

      return
    }

    // No match: pi lists everything with what was typed already in the search.
    showLoginProviders(undefined, typed.trim())
  }

  // ── pi's own /logout ───────────────────────────────────────────────

  /**
   * pi's "Select provider to logout:" — every credential this machine holds.
   *
   * A sign-in and a stored key both: pi's own list is what its auth store
   * holds, and its logout removes the credential and touches nothing the
   * provider declares. A key that comes from the environment is not stored and
   * is not listed; the row says which variable supplies it. pi's own sentence
   * for an empty list points at `/login`.
   */
  function showLogoutProviders(message?: string): void {
    const signedIn = providers
      .filter(item => item.authenticated && (item.auth_type === 'oauth' || !item.key_env))
      .sort((a, b) => a.name.localeCompare(b.name))

    const list = new SelectorList({
      emptyText: 'No providers logged in. Use /login first.',
      keybindings: deps.keybindings,
      onCancel: () => lease.done('cancel'),
      onSelect: slug => void logout(slug),
      placeholder: 'search providers',
      rows: signedIn.map(item => ({
        description: `${item.slug}  ·  ${item.auth_type === 'oauth' ? 'signed in' : 'API key'}`,
        id: item.slug,
        kind: 'item' as const,
        label: item.name,
        searchText: `${item.name} ${item.slug}`
      })),
      theme,
      title: 'Select provider to logout:',
      tui: deps.tui
    })

    if (message) {
      list.setError(message)
    }

    stage.set(list)
    deps.tui.requestRender()
  }

  /** Forget one stored credential, and say so the way pi says it. */
  async function logout(slug: string): Promise<void> {
    const name = provider(slug)?.name ?? slug

    try {
      const result = await deps.gateway.modelLogout({ session_id: sessionId, slug })

      if (!lease.isCurrent()) {
        return
      }

      if (!result.forgotten) {
        showLogoutProviders(`nothing is stored for ${name}`)

        return
      }

      // pi's own sentence for a removed grant.
      deps.transcript.print(`Logged out of ${name}`)
      lease.done('done')
    } catch (err) {
      if (!lease.isCurrent()) {
        return
      }

      // The gateway's own sentence: a sign-out it could not do says why.
      showLogoutProviders(stages.refusedBecause(err))
    }
  }

  // ── boot ───────────────────────────────────────────────────────────

  const loading = new SelectorList({
    keybindings: deps.keybindings,
    onCancel: () => lease.done('cancel'),
    onSelect: () => {},
    rows: [],
    theme,
    title: entry === 'login' ? 'Select authentication method:' : 'Select provider to logout:',
    tui: deps.tui
  })

  loading.setStatus('loading providers…')
  stage.set(loading)

  void loadOptions()
    .then(() => {
      if (!lease.isCurrent()) {
        return
      }

      if (entry === 'logout') {
        showLogoutProviders()
      } else if (entryProvider?.trim()) {
        enterNamedProvider(entryProvider)
      } else {
        showAuthMethodMenu()
      }
    })
    .catch((err: unknown) => {
      if (!lease.isCurrent()) {
        return
      }

      loading.setStatus(undefined)
      loading.setError(explain(err))
      deps.tui.requestRender()
    })

  return {
    component: stage,
    dispose: () => {
      // A mutation already sent is not rolled back by closing the picker; the
      // sequence guard is what stops its answer from painting a newer screen.
      sequence += 1
      // Whatever stage is up releases what it holds — a half-typed key, say.
      stage.current?.dispose?.()
    },
    focus: stage
  }
}
