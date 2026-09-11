// Auto-resume: wake the agent back up after retryable provider failures.
//
// The free-tier model endpoint intermittently returns 503 mid-stream. When
// that happens opencode ends the run and waits for human input. This plugin
// listens for those failures on the session.error bus event and re-submits a
// continuation nudge ("?") into the TUI, with escalating backoff and hard
// caps so it can never loop forever or fight the user.
//
// Only fires on APIError with isRetryable === true (503 bursts, timeouts).
// Never fires for MessageAbortedError (user pressed esc), ProviderAuthError
// (bad key — nudging cannot help), or MessageOutputLengthError.
//
// Tunables via env:
//   OPENCODE_AUTORESUME_MAX_CONSECUTIVE (default 5)
//   OPENCODE_AUTORESUME_MAX_PER_HOUR    (default 12)

import type { Plugin } from "@opencode-ai/plugin"

const COOLDOWN_BASE_MS = 15_000
const COOLDOWN_MAX_MS = 120_000

const MAX_CONSECUTIVE = Number(process.env.OPENCODE_AUTORESUME_MAX_CONSECUTIVE ?? 5)
const MAX_PER_HOUR = Number(process.env.OPENCODE_AUTORESUME_MAX_PER_HOUR ?? 12)

type ErrLike = {
  name?: string
  data?: { isRetryable?: boolean; message?: string; statusCode?: number }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

export default (async ({ client }) => {
  let consecutive = 0
  let lastNudgeAt = 0
  const hourStamps: number[] = []

  return {
    event: async ({ event }: { event: { type: string; properties?: Record<string, unknown> } }) => {
      try {
        // Successful work resets the escalation ladder.
        if (event.type === "session.status") {
          const status = (event.properties as { status?: { type?: string } } | undefined)?.status
          if (status?.type === "busy" && Date.now() - lastNudgeAt > 5_000) consecutive = 0
          return
        }
        if (event.type === "session.idle") {
          if (Date.now() - lastNudgeAt > 5_000) consecutive = 0
          return
        }
        if (event.type !== "session.error") return

        const err: ErrLike | undefined = (event.properties as { error?: ErrLike } | undefined)?.error
        if (!err || err.name !== "APIError" || err.data?.isRetryable !== true) return

        const now = Date.now()
        while (hourStamps.length && now - hourStamps[0] > 3_600_000) hourStamps.shift()
        if (hourStamps.length >= MAX_PER_HOUR) return
        const cooldown = Math.min(COOLDOWN_BASE_MS * 2 ** consecutive, COOLDOWN_MAX_MS)
        if (now - lastNudgeAt < cooldown) return
        if (consecutive >= MAX_CONSECUTIVE) return

        consecutive += 1
        lastNudgeAt = now
        hourStamps.push(now)

        console.log(
          `[auto-resume] retryable provider error #${consecutive} ` +
            `(statusCode=${err.data?.statusCode ?? "?"}); nudging in ${Math.round(cooldown / 1000)}s`
        )
        await sleep(cooldown)
        await client.tui.appendPrompt({ body: { text: "?" } })
        await client.tui.submitPrompt({})
      } catch {
        // Never let the watchdog take the host down.
      }
    },
  }
}) satisfies Plugin
