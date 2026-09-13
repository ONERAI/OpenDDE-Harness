// The credential, declare-an-endpoint, which-way-in and sign-in stages of
// pi's `/login`.
//
// They live beside the stage machine rather than inside it: the selector is
// the two doors and the two lists, and this is the part that talks to a
// provider — a key, an address, or pi's own sign-in flow.
//
// Everything they need from the selector arrives in `ModelStageContext` — the
// provider rows as last loaded, the way back to the provider list, and the way
// out once a provider is connected. They never reach into the selector's own
// state.

import { randomUUID } from 'node:crypto'

import type { FormFieldSpec } from '../components/formView.js'
import type { SelectorLease } from '../components/selector.js'
import type { StageView } from '../components/stage.js'
import type { LoginStepPush } from '../gateway.js'
import type { ModelOptionProvider } from '../rpc/index.js'
import type { SelectorDeps } from './deps.js'

import { FormView } from '../components/formView.js'
import { LoginView } from '../components/loginView.js'
import { validateSecret } from '../components/secretInput.js'
import { SelectorList } from '../components/selectorList.js'
import { ConfigValidationError } from '../rpc/index.js'

export interface ModelStageContext {
  /** Back to pi's provider list, with a sentence to show on it when there is one. */
  back(message?: string): void
  deps: SelectorDeps
  /** What an RPC failure should say, without the gateway's error payload. */
  explain(err: unknown): string
  /** The provider is connected: print pi's sentence and close the selector. */
  finish(message: string): void
  lease: SelectorLease
  /** The provider row as last loaded. */
  provider(slug: string): ModelOptionProvider | undefined
  sessionId: null | string
  stage: StageView
}

export interface ModelStages {
  /** What a refusal should say: the gateway's own sentence when it sent one. */
  refusedBecause(err: unknown): string
  /** pi's own "Select authentication method:" step, for a provider that offers
   *  both ways in. One way in goes straight to it, as pi does. */
  showAuthMethod(slug: string, onCancel: () => void): void
  /** The four fields a provider this config declares needs, asked at once. */
  showDeclareEndpoint(): void
  showKeyForm(slug: string, message?: string): void
  /** Runs pi's own sign-in here and now; there is nothing to confirm first. */
  showLogin(slug: string): void
}

/** pi's own generic label for the subscription option, used where the provider
 *  carries no `loginLabel` of its own (`interactive-mode.showLoginAuthTypeSelector`). */
export const SIGN_IN_WITH_ACCOUNT = 'Sign in with an account'
/** pi's own label for the key option. Not per provider: pi uses this one string. */
export const SIGN_IN_WITH_API_KEY = 'Sign in with an API key'

export function createModelStages(ctx: ModelStageContext): ModelStages {
  const { deps, explain, lease, sessionId, stage } = ctx
  const { theme } = deps

  const provider = (slug: string) => ctx.provider(slug)
  const back = (message?: string) => ctx.back(message)
  const finish = (message: string) => ctx.finish(message)

  function refusedBecause(err: unknown): string {
    if (err instanceof ConfigValidationError) {
      const detail = (err.data as { detail?: unknown } | null)?.detail

      if (typeof detail === 'string' && detail.trim()) {
        return detail
      }
    }

    return explain(err)
  }

  function showDeclareEndpoint(): void {
    const form = new FormView({
      fields: [
        {
          hint: 'what the entry is filed under, and the prefix of every model id it serves',
          id: 'provider',
          kind: 'text',
          label: 'Provider id',
          placeholder: 'my-vllm'
        },
        {
          hint: 'where the endpoint lives',
          id: 'base_url',
          kind: 'text',
          label: 'Base URL',
          placeholder: 'http://127.0.0.1:8000/v1'
        },
        {
          hint: 'optional: a server you run yourself usually wants none',
          id: 'api_key',
          kind: 'secret',
          label: 'API key',
          placeholder: 'paste the key'
        },
        {
          // The list is the endpoint's own, read from its /models the way pi
          // reads OpenRouter's; ids are typed only for one that publishes none.
          hint: 'optional: leave empty to read the endpoint’s own list; comma-separated otherwise',
          id: 'model',
          kind: 'text',
          label: 'Model ids',
          placeholder: 'discovered from GET /models'
        }
      ],
      keybindings: deps.keybindings,
      onCancel: () => {
        form.clearSecrets()
        back()
      },
      onSubmit: values => void declareEndpoint(values, form),
      submitLabel: 'Declare',
      // The wire is not asked for: this declares the one an arbitrary relay or a
      // self-hosted server implements. Another is written with `ddeharness
      // provider set`, which takes every field the entry has.
      subtitle:
        'a relay, a gateway or a server you run yourself, speaking Chat Completions — like OpenRouter, with your address and key',
      theme,
      title: 'Add an OpenAI-compatible endpoint'
    })

    stage.set(form)
    deps.tui.requestRender()
  }

  async function declareEndpoint(values: Record<string, string>, form: FormView): Promise<void> {
    const checked = validateSecret(values.api_key ?? '')

    if ('error' in checked) {
      form.setError(checked.error)
      deps.tui.requestRender()

      return
    }

    // Trimmed here and judged there: which ids and addresses are writable is the
    // gateway's answer, and a second copy of it would be one to drift from.
    const slug = (values.provider ?? '').trim()
    const baseUrl = (values.base_url ?? '').trim()
    const model = (values.model ?? '').trim()

    form.setBusy(true)
    form.setError(undefined)
    deps.tui.requestRender()

    try {
      const result = await deps.gateway.modelDeclareProvider({
        api_key: checked.value,
        base_url: baseUrl,
        model,
        provider: slug,
        session_id: sessionId
      })

      form.clearSecrets()

      if (!lease.isCurrent()) {
        return
      }

      const count = result.provider.models.length
      finish(`Logged in to ${result.provider.name} (${count} model${count === 1 ? '' : 's'} from ${baseUrl})`)
    } catch (err) {
      form.clearSecrets()

      if (!lease.isCurrent()) {
        return
      }

      form.setBusy(false)
      form.setError(refusedBecause(err))
      deps.tui.requestRender()
    }
  }

  /** Keep a mutation's row for its models, but not for its current flags: those
   *  describe the global selection, not this session's. */
  // ── key form ───────────────────────────────────────────────────────

  function showKeyForm(slug: string, message?: string): void {
    const item = provider(slug)
    // A provider the config *declares* is reached by an address, and an address
    // needs the wire it serves — both, or neither. One of pi's own carries both
    // itself, so the form does not ask and the gateway would refuse them.
    const declared = item?.needs_base_url ?? false
    const fields: FormFieldSpec[] = [
      {
        id: 'api_key',
        kind: 'secret',
        label: 'API key',
        placeholder: 'paste the key',
        // A redacted placeholder is never prefilled: submitting it would store
        // the mask as the key.
        hint: item?.key_env
          ? `${item.key_env} is already set — leave this blank to use it`
          : item?.needs_api_key
            ? 'required for this provider'
            : 'optional for this provider'
      }
    ]

    if (declared) {
      fields.push(
        {
          hint: 'where this provider lives',
          id: 'base_url',
          kind: 'text',
          label: 'Base URL',
          value: item?.base_url ?? ''
        },
        {
          // Not an enum: the wires are the model service's own list, and a copy
          // of it here would be a second answer to drift from. A value it does
          // not implement comes back refused, naming every one it does.
          hint: 'the wire that address serves, e.g. openai-completions',
          id: 'api',
          kind: 'text',
          label: 'API',
          value: item?.api ?? ''
        }
      )
    }

    const form = new FormView({
      fields,
      keybindings: deps.keybindings,
      onCancel: () => {
        form.clearSecrets()
        back()
      },
      onSubmit: values => void saveKey(slug, values, form),
      submitLabel: 'Save',
      subtitle: declared
        ? 'this provider is declared here: its address, its wire, and a key only if it wants one'
        : item?.authenticated
          ? 'this provider already has a key; saving replaces it'
          : 'the key is stored by the gateway, for this provider',
      theme,
      title: `Connect ${item?.name ?? slug}`
    })

    if (message) {
      form.setError(message)
    }

    stage.set(form)
    deps.tui.requestRender()
  }

  async function saveKey(slug: string, values: Record<string, string>, form: FormView): Promise<void> {
    const item = provider(slug)
    const checked = validateSecret(values.api_key ?? '')

    if ('error' in checked) {
      form.setError(checked.error)
      deps.tui.requestRender()

      return
    }

    const baseUrl = (values.base_url ?? '').trim()
    const api = (values.api ?? '').trim()

    if (item?.needs_api_key && !checked.value) {
      form.setError('this provider needs an API key')
      deps.tui.requestRender()

      return
    }

    if (item?.needs_base_url && !baseUrl) {
      form.setError('this provider needs a base URL')
      deps.tui.requestRender()

      return
    }

    if (item?.needs_base_url && !api) {
      form.setError('an address needs the wire it serves — say which')
      deps.tui.requestRender()

      return
    }

    form.setBusy(true)
    form.setError(undefined)
    deps.tui.requestRender()

    try {
      const result = await deps.gateway.modelSaveKey({
        api_key: checked.value,
        session_id: sessionId,
        slug,
        ...(baseUrl ? { base_url: baseUrl } : {}),
        ...(api ? { api } : {})
      })

      form.clearSecrets()

      if (!lease.isCurrent()) {
        return
      }

      if (result.provider.authenticated) {
        // pi's own sentence once a credential is stored.
        finish(`Logged in to ${result.provider.name}`)

        return
      }

      form.setBusy(false)
      form.setError(result.provider.warning || 'the gateway did not accept that credential')
      deps.tui.requestRender()
    } catch (err) {
      form.clearSecrets()

      if (!lease.isCurrent()) {
        return
      }

      form.setBusy(false)
      form.setError(explain(err))
      deps.tui.requestRender()
    }
  }

  // ── which way in ───────────────────────────────────────────────────

  /**
   * pi's own authentication-method step.
   *
   * The menu is pi's, down to the strings: the subscription option is the
   * provider's own `oauth.loginLabel` where pi wrote one and pi's generic
   * "Sign in with an account" where it did not, the key option is always
   * "Sign in with an API key", and the subscription comes first. A provider with
   * one way in is not asked — pi goes straight to it, and so does this.
   *
   * `auth_methods` is pi's own list, read off its provider objects by the model
   * service. A gateway that could not ask it sends none, and then the row's own
   * `auth_type` decides, which is what happened before pi was asked at all.
   */
  function showAuthMethod(slug: string, onCancel: () => void): void {
    const item = provider(slug)
    const methods = item?.auth_methods ?? []
    const key = () => showKeyForm(slug)

    if (methods.length < 2) {
      const only = methods[0] ?? item?.auth_type

      if (only === 'oauth') {
        showLogin(slug)
      } else {
        key()
      }

      return
    }

    const list = new SelectorList({
      keybindings: deps.keybindings,
      onCancel,
      onSelect: id => (id === 'oauth' ? showLogin(slug) : key()),
      placeholder: 'search methods',
      rows: [
        {
          // pi's own sentence for this provider when it has one.
          description: item?.login_label ? slug : `${slug}  ·  subscription`,
          id: 'oauth',
          kind: 'item',
          label: item?.login_label || SIGN_IN_WITH_ACCOUNT,
          searchText: `${SIGN_IN_WITH_ACCOUNT} ${item?.login_label ?? ''} oauth subscription account`
        },
        {
          description: item?.key_label ? `${slug}  ·  ${item.key_label}` : `${slug}  ·  API key`,
          id: 'key',
          kind: 'item',
          label: SIGN_IN_WITH_API_KEY,
          searchText: `${SIGN_IN_WITH_API_KEY} api key`
        }
      ],
      theme,
      // pi titles this step with the provider's name once it knows which one.
      title: `Select authentication method for ${item?.name ?? slug}:`,
      tui: deps.tui
    })

    stage.set(list)
    deps.tui.requestRender()
  }

  // ── sign-in ────────────────────────────────────────────────────────

  /**
   * Show one step that is only there to be read, the way pi's login dialog shows
   * it (`login-dialog.showDeviceCode` / `showAuth` / `showInfo` / `showProgress`).
   *
   * pi's lines, in pi's order, with pi's words: the URL on its own line as a
   * hyperlink, the click hint under it, then the code as "Enter code: <code>"
   * and the wait pi prints after it. Nothing here writes a sentence of its own —
   * every string comes from the event or from pi's own dialog.
   */
  function showStep(view: LoginView, step: LoginStepPush['step']): void {
    if (step.type === 'device_code') {
      view.addLink(step.verificationUri)
      view.addLine(`Enter code: ${step.userCode}`, 'warn')
      view.setStatus('Waiting for authentication...')

      return
    }

    if (step.type === 'auth_url') {
      view.addLink(step.url)

      if (step.instructions) {
        view.addLine(step.instructions, 'warn')
      }

      // pi's dialog opens the browser for this step (`showAuth`) and not for a
      // device code, whose URL is meant for another machine.
      deps.openBrowser(step.url)

      return
    }

    if (step.type === 'info') {
      view.addLine(step.message)

      for (const link of step.links ?? []) {
        view.addLink(link.url, link.label)
      }

      return
    }

    view.addLine(step.message, 'muted')
  }

  /**
   * Run the sign-in here, showing pi's own flow as it happens.
   *
   * The gateway holds the flow (`model.login`) and pushes every step it
   * reports; the two it stops on are answered from this view — the login-method
   * menu with pi's own labels, and the authorization code a browser login falls
   * back to asking for. The request settles when a credential is stored, which
   * is as long as the person takes, so what ends it early is the cancel this
   * view sends.
   */
  function showLogin(slug: string): void {
    const item = provider(slug)
    /**
     * The id this flow's pushed steps carry, chosen here so it is known before
     * the first step arrives: another sign-in's steps -- an earlier one for
     * the same provider that was left on Esc -- carry a different one and are
     * not this view's to show, and a cancel can name the flow at any moment.
     */
    const loginId = randomUUID()
    /** Esc was pressed, or the view went away: the flow is not wanted any more. */
    let cancelled = false
    /** The request settled, one way or the other. */
    let settled = false
    let unsubscribe = () => {}

    const settle = () => {
      settled = true
      unsubscribe()
      unsubscribe = () => {}
    }

    // Best effort: the stage is leaving either way, and a login that has already
    // ended answers false.
    const sendCancel = () => void deps.gateway.modelLoginCancel({ login_id: loginId }).catch(() => {})

    /** End the flow at the gateway, once, however this view is left. */
    const abandon = () => {
      if (cancelled || settled) {
        return
      }

      cancelled = true
      sendCancel()
      settle()
    }

    const cancel = () => {
      abandon()
      back()
    }

    const view = new LoginView({
      keybindings: deps.keybindings,
      onAnswer: answer => {
        view.clearAsk()
        view.setStatus('working…')
        deps.tui.requestRender()
        void deps.gateway.modelLoginAnswer({ answer, login_id: loginId }).catch((err: unknown) => {
          if (lease.isCurrent()) {
            view.setError(explain(err))
            deps.tui.requestRender()
          }
        })
      },
      onCancel: cancel,
      // Replaced by another stage, or the picker closed over it: nobody is
      // watching, so the vendor is not left polling for a code.
      onDispose: abandon,
      subtitle: 'this signs in for the whole machine, not just this conversation',
      theme,
      title: `Sign in to ${item?.name ?? slug}`
    })

    view.setStatus('starting the sign-in…')

    unsubscribe = deps.onLoginStep(push => {
      if (push.login_id !== loginId || cancelled || !lease.isCurrent()) {
        return
      }

      const { step } = push

      if (step.type === 'select') {
        view.setStatus(undefined)
        view.askSelect(step.message, step.options)
      } else if (step.type === 'manual_code') {
        view.setStatus(undefined)
        view.askText(step.message, step.placeholder)
      } else {
        // A step to read, not to answer: the menu is behind us, and a line that
        // arrives while a question is up belongs after it.
        view.setStatus(undefined)
        showStep(view, step)
      }

      deps.tui.requestRender()
    })

    stage.set(view)
    deps.tui.requestRender()
    void runLogin(slug, loginId, () => cancelled, settle)
  }

  async function runLogin(
    slug: string,
    loginId: string,
    wasCancelled: () => boolean,
    settle: () => void
  ): Promise<void> {
    try {
      const result = await deps.gateway.modelLogin({ login_id: loginId, provider: slug, session_id: sessionId })

      settle()

      if (!lease.isCurrent() || wasCancelled()) {
        return
      }

      // pi's own sentence for a finished sign-in, and the selector closes on
      // it the way pi's does.
      finish(`Logged in to ${result.provider.name}`)
    } catch (err) {
      settle()

      if (!lease.isCurrent() || wasCancelled()) {
        return
      }

      // Back to the list rather than back to this stage: entering the provider
      // is what starts a sign-in, so re-showing the stage would start another
      // one nobody asked for. pi's own sentence for a failed one, with the
      // gateway's own detail after it.
      back(`Failed to login to ${provider(slug)?.name ?? slug}: ${refusedBecause(err)}`)
    }
  }

  return {
    refusedBecause,
    showAuthMethod,
    showDeclareEndpoint,
    showKeyForm,
    showLogin
  }
}
