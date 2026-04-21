import { useState, useEffect, useRef, useCallback } from 'react'

/**
 * Polls a fetcher function every `intervalMs` milliseconds.
 * Returns { data, error, loading, refresh }.
 */
export function usePolling(fetcher, intervalMs = 5000, enabled = true) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const timerRef = useRef(null)
  const mountedRef = useRef(true)

  const refresh = useCallback(async () => {
    if (!mountedRef.current) return
    try {
      const result = await fetcher()
      if (mountedRef.current) {
        setData(result)
        setError(null)
      }
    } catch (err) {
      if (mountedRef.current) setError(err.message)
    } finally {
      if (mountedRef.current) setLoading(false)
    }
  }, [fetcher])

  useEffect(() => {
    mountedRef.current = true
    if (!enabled) { setLoading(false); return }

    refresh()
    if (intervalMs > 0) {
      timerRef.current = setInterval(refresh, intervalMs)
    }
    return () => {
      mountedRef.current = false
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [enabled, intervalMs, refresh])

  return { data, error, loading, refresh }
}
