import { useLocation } from 'react-router-dom'
import { Activity } from 'lucide-react'
import { useState, useEffect } from 'react'

const ROUTE_TITLES = {
  '/': 'Dashboard',
  '/arena': 'Episode Arena',
  '/tasks': 'Task Bank',
  '/forge': 'Forge Queue',
  '/oversight': 'Oversight Stats',
  '/difficulty': 'Difficulty Curve',
  '/grader': 'Standalone Grader',
}

export default function Header() {
  const { pathname } = useLocation()
  const title = ROUTE_TITLES[pathname] || 'Forge Arena'
  const [serverOk, setServerOk] = useState(null)
  const [time, setTime] = useState(new Date())

  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch('/api/forge/queue')
        setServerOk(r.ok)
      } catch {
        setServerOk(false)
      }
    }
    check()
    const t = setInterval(check, 10000)
    return () => clearInterval(t)
  }, [])

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  return (
    <header className="h-14 border-b border-border bg-panel flex items-center px-6 sticky top-0 z-20">
      <div className="flex-1">
        <h1 className="text-base font-semibold text-primary">{title}</h1>
      </div>
      <div className="flex items-center gap-5">
        <span className="text-xs font-mono text-muted">{time.toLocaleTimeString()}</span>
        <div className="flex items-center gap-1.5 text-xs font-mono">
          <Activity size={12} className={serverOk === null ? 'text-muted' : serverOk ? 'text-green-neon' : 'text-red-neon'} />
          <span className={serverOk === null ? 'text-muted' : serverOk ? 'text-green-neon' : 'text-red-neon'}>
            {serverOk === null ? 'checking' : serverOk ? 'online' : 'offline'}
          </span>
        </div>
      </div>
    </header>
  )
}
