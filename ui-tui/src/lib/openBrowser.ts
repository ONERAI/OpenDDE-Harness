// pi's own browser launcher (`coding-agent/src/utils/open-browser.ts`), which
// its login dialog calls when a sign-in hands over a URL. Copied rather than
// depended on: the coding agent is not a package this project installs.
//
// Never through a shell. On Windows `cmd /c start` would re-parse the URL's
// metacharacters before `start` saw it, which makes a vendor-supplied URL
// injectable; `rundll32` takes it as one argument.

import { spawn } from 'node:child_process'

/** Open a URL in the platform's default browser. Best effort: the URL is on
 *  screen either way, so a launcher that is not there is not an error. */
export function openBrowser(target: string): void {
  const [cmd, args]: [string, string[]] =
    process.platform === 'darwin'
      ? ['open', [target]]
      : process.platform === 'win32'
        ? ['rundll32', ['url.dll,FileProtocolHandler', target]]
        : ['xdg-open', [target]]

  spawn(cmd, args, { detached: true, stdio: 'ignore' })
    .on('error', () => {})
    .unref()
}
